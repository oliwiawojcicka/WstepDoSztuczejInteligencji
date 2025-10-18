import numpy as np
from algorytm import calc_target, algorytm_genetyczny
from matplotlib import pyplot as plt
from genetic_solver import GeneticSolver

def q(x):
    return calc_target(np.array(x))

def test_best_hyperparams():
    """
    Testuje zestaw znalezionych hiperparametrów dla 10 losowych random seedów.
    Celem jest sprawdzenie, czy dla każdego seeda algorytm osiąga stabilny wynik
    (funkcja celu powyżej -4 oznacza skuteczne sterowanie rakietą).
    """
    # Losujemy 10 różnych wartości ziarna, żeby sprawdzić odporność na losowość
    seeds = np.random.randint(0, 100, size=10)
    print("----- Wylosowane seedy -----")
    print(seeds)
    # Wybrany zestaw hiperparametrów
    mu = 80
    pm = 0.001
    pc = 0.9
    t_max = 120
    n_genes = 400
    results = [] # lista do przechowywania wyników funkcji celu dla każdego seeda
    print("\n----- Wyniki dla 10 random seedów -----")
    for seed in seeds:
        # Dla każdego seeda tworzymy nowy solver z tym samym zestawem hiperparametrów,ale inną losową populacją startową
        solver = GeneticSolver(mu, pm, pc, t_max, n_genes, seed)
        best_x, best_s = solver.solve(q)
        # Zapisujemy wartość funkcji celu
        results.append(best_s)
        print(f"Seed={seed:<5}  ->  wynik={best_s:8.8f}")
    results = np.array(results)
    # Liczymy, ile przypadków zakończyło się sukcesem (wynik powyżej -4)
    good = np.sum(results > -4)
    print("\n----- PODSUMOWANIE -----")
    print(f"Liczba przypadków z wynikiem > -4: {good}/10")
    print(f"Średni wynik:   {np.mean(results):8.8f}")
    print(f"Odchylenie std: {np.std(results):8.8f}")
    print(f"Najlepszy wynik:{np.max(results):8.8f}")
    print(f"Najgorszy wynik:{np.min(results):8.8f}")
    return seeds, results

def test_influence_of_mu():
    """
    Analizuje wpływ liczby osobników (μ) na wynik działania algorytmu.
    Pozwala zrozumieć, jak rozmiar populacji wpływa na stabilność i dokładność ewolucji.
    """
    mu_values = [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240, 280, 320, 360, 400]
    pm = 0.001
    pc = 0.9
    t_max = 120
    n_genes = 400
    seed = 25
    # Listy do przechowywania wyników funkcji celu i błędu końcowego
    results_q = []
    results_error = []
    print("\n----- WPŁYW PARAMETRU μ -----")
    for mu in mu_values:
        # Uruchamiamy solver z różną liczbą osobników
        solver = GeneticSolver(mu, pm, pc, t_max, n_genes, seed)
        best_x, best_s = solver.solve(q)
        blad = np.sqrt(abs(best_s))
        results_q.append(best_s)
        results_error.append(blad)
        print(f"μ={mu:<4}  -> wynik={best_s:10.7f}   błąd≈{blad:8.4f} m")
    # Wykres pozwala wizualnie ocenić trend: czy większe μ poprawia wynik i stabilność
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = 'tab:blue'
    ax1.set_xlabel('μ (liczba osobników)')
    ax1.set_ylabel('Wartość funkcji celu q(x)', color=color)
    ax1.plot(mu_values, results_q, marker='o', color=color, label='Funkcja celu q')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    plt.savefig("wplyw_mu6.png")
    plt.show()
    return mu_values, results_q, results_error



