import numpy as np
from algorytm import calc_target, algorytm_genetyczny
from matplotlib import pyplot as plt
from genetic_solver import GeneticSolver

def q(x):
    return calc_target(np.array(x))

# Losujemy 10 różnych wartości ziarna, żeby sprawdzić odporność na losowość
seeds = np.random.randint(0, 100, size=10)

def test_best_hyperparams():
    """
    Testuje zestaw znalezionych hiperparametrów dla 10 losowych random seedów.
    Celem jest sprawdzenie, czy dla każdego seeda algorytm osiąga stabilny wynik
    (funkcja celu powyżej -4 oznacza skuteczne sterowanie rakietą).
    """
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
    Analizuje wpływ liczby osobników (μ) na wynik działania algorytmu genetycznego.
    Dla każdej wartości μ testuje 10 różnych seedów, a następnie oblicza średni wynik i zmienność.
    """
    mu_values = [10, 20, 30, 40, 50, 60, 80, 100, 120, 160, 200, 240] #za dużo prowadzi do bardzo długiego czasu obliczeń
    # Stałe hiperparametry
    pm = 0.001
    pc = 0.9
    t_max = 120
    n_genes = 400
    avg_results_q = []      # średnia wartość funkcji celu dla każdego μ
    std_results_q = []      # odchylenie standardowe funkcji celu
    avg_results_error = []  # średni błąd
    std_results_error = []  # odchylenie standardowe błędu
    print("\n----- WPŁYW PARAMETRU μ (średnia z 10 seedów) -----")
    for mu in mu_values:
        results_q = []
        results_error = []
        for seed in seeds:
            # Uruchamiamy solver z różną liczbą osobników
            solver = GeneticSolver(mu, pm, pc, t_max, n_genes, seed)
            best_x, best_s = solver.solve(q)
            blad = np.sqrt(abs(best_s))
            results_q.append(best_s)
            results_error.append(blad)
        # Obliczanie średnich i odchyleń
        avg_q = np.mean(results_q)
        std_q = np.std(results_q)
        avg_err = np.mean(results_error)
        std_err = np.std(results_error)
        avg_results_q.append(avg_q)
        std_results_q.append(std_q)
        avg_results_error.append(avg_err)
        std_results_error.append(std_err)
        print(f"μ={mu:<4}  →  ŚREDNI wynik={avg_q:10.6f} ± {std_q:8.6f}   "
              f"ŚREDNI błąd={avg_err:8.4f} ± {std_err:6.4f} m")
    # Wykres pozwala wizualnie ocenić trend: czy większe μ poprawia wynik i stabilność
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel('μ (liczba osobników)')
    ax1.set_ylabel('Średnia wartość funkcji celu q(x)')
    ax1.errorbar(mu_values, avg_results_q, yerr=std_results_q, fmt='-o', color='tab:blue', ecolor='lightgray', elinewidth=2, capsize=4)
    ax1.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("wplyw_mu_avg.png")
    plt.show()
    return mu_values, avg_results_q, std_results_q, avg_results_error, std_results_error

