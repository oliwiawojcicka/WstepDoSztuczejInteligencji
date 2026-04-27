import numpy as np
from solver import Solver
from algorytm import algorytm_genetyczny, calc_target

class GeneticSolver(Solver):
    """Solver wykorzystujący algorytm genetyczny."""

    def __init__(self, mu, pm, pc, t_max, n_genes, seed):
        self.mu = mu           # liczba osobników (rozmiar populacji)
        self.pm = pm           # prawdopodobieństwo mutacji
        self.pc = pc           # prawdopodobieństwo krzyżowania
        self.t_max = t_max     # liczba generacji (iteracji)
        self.n_genes = n_genes # liczba genów (czyli długość sterowania)
        np.random.seed(seed)   # ziarno losowości (żeby wyniki były powtarzalne)

    def get_parameters(self):
        return {
            'mu': self.mu,
            'pm': self.pm,
            'pc': self.pc,
            't_max': self.t_max,
            'n_genes': self.n_genes
        }

    def solve(self, q):
        """
        Uruchamia klasyczny algorytm genetyczny i zwraca:
        - najlepsze znalezione rozwiązanie (best_x),
        - jego wartość funkcji celu (best_s).
        """
        P0 = np.random.randint(0, 2, (self.mu, self.n_genes))
        best_x, best_s = algorytm_genetyczny(
            q, P0, self.mu, self.pm, self.pc, self.t_max
        )
        return best_x, best_s