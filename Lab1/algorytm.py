import numpy as np

def calc_path(control: np.ndarray, time: int = 200, dt: float = 0.1):
    """Symulacja ruchu obiektu sterowanego wektorem 'control'."""
    pathx = []  # poruszanie poziome (historia położenia X)
    pathz = []  # poruszanie pionowe (historia położenia Z)
    # Inicjalizacja pozycji i prędkości (rakieta startuje z punktu (0,0))
    posx = np.zeros(control.shape[:-1])
    posz = np.zeros(control.shape[:-1])
    velx = np.zeros(control.shape[:-1])
    velz = np.zeros(control.shape[:-1])
    # Każdy osobnik ma 2 sterowania (silnik X i silnik Z) dla każdego kroku czasu
    control = control.reshape(*control.shape[:-1], time, 2)
    t = 0
    pathx.append(posx)
    pathz.append(posz)
    # Symulujemy dopóki którakolwiek rakieta nie spadnie poniżej ziemi (posz < 0)
    while (posz >= 0).any():
        if t < time:
            cx = control[..., t, 0]
            cz = control[..., t, 1]
        else:
            cx = 0
            cz = 0
        # Aktualizacja prędkości z uwzględnieniem oporów i grawitacji
        velx = velx + (cx * 15 - 0.5 * velx) * dt
        velz = velz + (cz * 15 - 9.8 - 0.5 * velz) * dt
        # Gdy rakieta spadnie (posz < 0), zatrzymujemy jej ruch
        velx = velx * (posz >= 0)
        velz = velz * (posz >= 0)
        # Aktualizacja położenia
        posx = posx + velx * dt
        posz = posz + velz * dt
        # Zapisujemy przebieg lotu
        pathx.append(posx)
        pathz.append(posz)
        t += 1
    return pathx, pathz

def calc_target(control: np.ndarray):
    """Funkcja celu – chcemy osiągnąć 350 m poziomo."""
    # Obliczamy pełną trajektorię dla danego przebiegu lotu
    pathx, pathz = calc_path(control)
    # Zwracamy ujemny kwadrat odległości od celu (350 m)
    # Ujemny, bo algorytm genetyczny jest maksymalizujący — im bliżej celu, tym wyższy wynik
    return -(pathx[-1] - 350) ** 2

#  ELEMENTY ALGORYTMU GENETYCZNEGO

def ocena(q, populacja):
    """Ocena osobników przez funkcję celu."""
    # Dla każdego osobnika obliczamy wartość funkcji celu q(x)
    return np.array([q(x) for x in populacja])

def znajdz_najlepszego(populacja, oceny):
    # Szukamy indeksu najlepszego osobnika – tutaj maksimum, bo maksymalizujemy q
    idx = np.argmax(oceny)
    return populacja[idx], oceny[idx]

def reprodukcja_ruletkowa(populacja, oceny, μ):
    # Wyniki mogą być ujemne – przesuwamy cały wektor, żeby był dodatni
    roznica = oceny - np.min(oceny) + 1e-6  # +1e-6 zapobiega dzieleniu przez zero
    # Normalizujemy
    stosunek = roznica / np.sum(roznica)
    # Losujemy μ osobników z powtórzeniami według rozkładu 'stosunek'
    # Dzięki temu lepsi mają większą szansę, ale słabsi też czasem przechodzą dalej
    indeksy = np.random.choice(len(populacja), size=μ, p=stosunek)
    return populacja[indeksy]

def krzyzowanie_jednopunktowe(p1, p2):
    n = len(p1)
    # Losujemy punkt przecięcia – unikamy 0 i n, żeby faktycznie wymieszać geny
    punkt = np.random.randint(1, n - 1)
    # Tworzymy dwa potomki przez wymianę fragmentów genotypu
    d1 = np.concatenate((p1[:punkt], p2[punkt:]))
    d2 = np.concatenate((p2[:punkt], p1[punkt:]))
    return d1, d2

def mutacja_bitowa(osobnik, pm):
    # True tam, gdzie zachodzi mutacja
    czy_mutacja = np.random.rand(len(osobnik)) < pm
    # Odwracamy wartości bitów tylko w tych pozycjach, które miały True
    osobnik[czy_mutacja] = 1 - osobnik[czy_mutacja]
    return osobnik

def krzyzowanie_i_mutacja(populacja, pm, pc):
    np.random.shuffle(populacja)  # losowo mieszamy, żeby za każdym razem inne osobniki były łączone w pary do krzyżowania
    nowa = []
    for i in range(0, len(populacja), 2):
        p1, p2 = populacja[i], populacja[(i + 1) % len(populacja)]
        # Krzyżowanie z prawdopodobieństwem pc
        if np.random.rand() < pc:
            d1, d2 = krzyzowanie_jednopunktowe(p1, p2)
        else:
            # Brak krzyżowania – dzieci to kopie rodziców
            d1, d2 = p1.copy(), p2.copy()
        # Po krzyżowaniu każde dziecko poddajemy mutacji z prawdopodobieństwem pm
        nowa.append(mutacja_bitowa(d1, pm))
        nowa.append(mutacja_bitowa(d2, pm))
    return np.array(nowa)

#  GŁÓWNY ALGORYTM GENETYCZNY
def algorytm_genetyczny(q, P0, μ, pm, pc, t_max):
    populacja = P0.copy()        # kopia początkowej populacji
    oceny = ocena(q, populacja)  # obliczenie wartości funkcji celu
    x_best, o_best = znajdz_najlepszego(populacja, oceny)
    for t in range(t_max):
        # Reprodukcja: tworzymy nową pulę rodziców wg ruletki
        R = reprodukcja_ruletkowa(populacja, oceny, μ)
        # Krzyżowanie i mutacja – tworzymy kolejne pokolenie
        M = krzyzowanie_i_mutacja(R, pm, pc)
        # Oceniamy nowe pokolenie
        oceny = ocena(q, M)
        # Szukamy najlepszego osobnika w tej generacji
        x_t, o_t = znajdz_najlepszego(M, oceny)
        # Jeśli znaleziono lepszego niż dotychczasowy globalny lider – zapamiętaj go
        if o_t > o_best:
            x_best, o_best = x_t, o_t
        # Następne pokolenie staje się aktualnym
        populacja = M
    # Po wykonaniu wszystkich pokoleń zwracamy najlepsze znalezione rozwiązanie
    return x_best, o_best


