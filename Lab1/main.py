import numpy as np
from algorytm import algorytm_genetyczny, calc_target
from test import test_best_hyperparams, test_influence_of_mu

def main():

    #---------------Trywialne i losowe rozwiązanie----------------
    lista = {
        "Wszystkie wyłączone (0,0)": np.zeros(200 * 2, dtype=int),
        "Wszystkie włączone (1,1)": np.ones(200 * 2, dtype=int),
        "Tylko poziome (1,0)": np.tile([1, 0], 200),
        "Tylko pionowe (0,1)": np.tile([0, 1], 200),
        "Losowe sterowanie": np.random.randint(0, 2, 200 * 2, dtype=int)
    }

    print("----- TRYWIALNE I LOSOWE STEROWANIA -----")
    for name, ctrl in lista.items():
        wynik = float(calc_target(ctrl))
        blad = np.sqrt(abs(wynik))
        print(f"{name:30s}  wynik={wynik:10.2f}   błąd≈{blad:7.2f} m")
    print()

    # ---------------Górne ograniczenie wyniku----------------
    print("----- Górne ograniczenie wyniku wynosi: 0 -----")
    print()


if __name__ == "__main__":
    main()
    test_best_hyperparams()
    test_influence_of_mu()