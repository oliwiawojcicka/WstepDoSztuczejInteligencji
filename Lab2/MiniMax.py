from __future__ import annotations
import math
import random
from typing import Callable, Any, List, Optional

Heuristic = Callable[[Any, Any], float]  #funkcja przyjmuje 2 argumenty i zwraca float

#---- Heurystyka gry ---
def heuristic(state, root_player) -> float:
    """
    Funkcja heurystyczna dla gry Szewc.
    Zwraca ocenę stanu z punktu widzenia gracza `root_player`:
      • Dodatnia wartość  → stan korzystny dla root_player
      • Ujemna wartość    → stan korzystny dla przeciwnika
      • 0                 → stan neutralny (remisowy)

      - W trakcie gry: różnica punktów (nasze - przeciwnika)
      - W stanie końcowym:
          +∞ jeśli root_player wygrał,
          −∞ jeśli przegrał,
           0 jeśli remis
    """
    scores = state.get_scores()
    our_score = scores.get(root_player, 0)  #punkty gracza, którego oceniamy (root_player)
                                            #0 to wartość domyślna – oznacza, że gracz ma 0 punktów, jeśli nie ma go jeszcze w słowniku wyników
    opponent_score = sum(scores.values()) - our_score  #punkty przeciwnika
    score_difference = our_score - opponent_score
    if state.is_finished():
        winner = state.get_winner()
        if winner is None:
            return 0.0  #remis
        if winner == root_player:
            return float('inf')  #nieskończenie dobra pozycja -> wygrana
        else:
            return float('-inf')  #nieskończenie zła pozycja -> przegrana
    return float(score_difference)

#--- MiniMax z obcięciem α–β ---
def minimax(state, depth: int, alpha: float, beta: float, is_maximizing_turn: bool, root_player, heuristic: Heuristic) -> float:
    """
    Algorytm Minimax z obcinaniem alfa–beta.
    Zwraca wartość oceny stanu z punktu widzenia gracza `root_player`.
    """
    if depth == 0 or state.is_finished():  #jeśli osiągnięto maksymalną głębokość lub koniec gry -> koniec rekurencji
        return heuristic(state, root_player)
    possible_moves = list(state.get_moves())  #wszystkie dostępne ruchy z bieżącego stanu
    if not possible_moves:                                 #jeśli gra jeszcze nie jest formalnie zakończona, ale żaden gracz nie ma aktualnie legalnych ruchów
        return heuristic(state, root_player)
    #maksymalizujący
    if is_maximizing_turn:
        best_value = -math.inf  #początkowo najgorsza możliwa wartość dla max
        for move in possible_moves:
            next_state = state.make_move(move) #wykonaje ruch, czyli uzyskaje stan potomny
                                               #rekurencyjnie ocenia stan potomny
            #sprawdzam, czy po tym ruchu znów gra nasz gracz (bo np. mógł zdobyć pole i w Szewcu ma dodatkowy ruch)
            value = minimax(
                next_state, depth - 1, alpha, beta,
                next_state.get_current_player() == root_player,  #określenie, kto teraz gra, czy min czy max
                root_player, heuristic
            )
            best_value = max(best_value, value)  #wybieramy lepszy wynik
            alpha = max(alpha, best_value)
            if best_value >= beta:  #warunek obcięcia (β-cutoff)
                break  #wiemy, że przeciwnik min nigdy pozwoli na wejście w tę gałąź wyżej (bo ma już lepszą/tańszą opcję) -> break
        return best_value  #zwraca najlepszy znaleziony wynik dla tej gałęzi
    #minimalizujący
    else:
        best_value = math.inf
        for move in possible_moves:
            next_state = state.make_move(move)
            value = minimax(
                next_state, depth - 1, alpha, beta,
                next_state.get_current_player() == root_player,
                root_player, heuristic
            )
            best_value = min(best_value, value)
            beta = min(beta, best_value)
            if best_value <= alpha:  #warunek obcięcia (α-cutoff)
                break
        return best_value

#--- Wybór ruchu w grze ---
def choose_move(state, depth: int, heuristic: Heuristic, rng: Optional[random.Random] = None, tolerance: float = 1e-9):
    """
    Wybiera najlepszy ruch przy użyciu minimaxa z obcinaniem alfa–beta.
    Zwraca: (najlepszy_ruch, wartość).
    Jeśli kilka ruchów ma taką samą wartość (z tolerancją), to wybiera losowo jeden z nich.
    """
    rng = rng or random.Random()  #generator losowy
    root_player = state.get_current_player()  #gracz rozpoczynający (max)
    best_value = -math.inf
    best_moves: List[Any] = []  #lista wszystkich ruchów, które dają tę samą najlepszą wartość (bo może być remis)
    alpha, beta = -math.inf, math.inf
    for move in state.get_moves():
        next_state = state.make_move(move)
        value = minimax(
            next_state, depth - 1, alpha, beta,
            next_state.get_current_player() == root_player,   #czy po ruchu gra nasz gracz
            root_player, heuristic
        )
        if value > best_value + tolerance:  #czy lepszy
            best_value, best_moves = value, [move]
        elif abs(value - best_value) <= tolerance:  #czy remis
            best_moves.append(move)
        alpha = max(alpha, best_value)
    if not best_moves:
        return None, heuristic(state, root_player)
    return rng.choice(best_moves), best_value #wylosuje ruch spośród najlepszych

