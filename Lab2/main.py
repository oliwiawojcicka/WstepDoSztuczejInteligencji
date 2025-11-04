# main.py
from __future__ import annotations
import random
import statistics
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
from dots_and_boxes import DotsAndBoxes
from MiniMax import choose_move, heuristic


#--- Pojedyncza gra z dla wizualizacji ---
def play_game(size: int, depth_a: int, depth_b: int, seed: int, verbose: bool) -> Tuple[int, object]:
    """
    Rozgrywa jedną partię Szewca.
    verbose=True → wypisuje planszę po każdym ruchu oraz punkty
    Zwraca (różnica A−B, zwycięzca lub remis).
    """
    rng = random.Random(seed)
    game = DotsAndBoxes(size=size)  #inicjalizacja gry
    if verbose:
        print(f"\n== START: board {size}x{size}, A-depth={depth_a}, B-depth={depth_b}, seed={seed} ==")
        print(game.state)
    move_counter = 0
    while not game.is_finished():
        state = game.state
        current_player = state.get_current_player()
        current_depth = depth_a if current_player == game.first_player else depth_b
        move, _ = choose_move(state, current_depth, heuristic, rng)  #wybór ruchu minimaxem
        if move is None:
            break
        move_counter += 1
        if verbose:
            print(f"\nRuch #{move_counter}: gracz {current_player.char}")
        game.make_move(move)  #wykonanie ruchu
        if verbose:
            print(game.state)  #plansza po ruchu
            scores = game.state.get_scores()
            score_a = scores.get(game.first_player, 0)
            score_b = scores.get(game.second_player, 0)
            print(f"Punkty: A={score_a}  B={score_b}")
    #podsumowanie
    winner = game.get_winner()
    scores = game.state.get_scores()
    score_a = scores.get(game.first_player, 0)
    score_b = scores.get(game.second_player, 0)
    if verbose:
        print("\n== KONIEC ==")
        print(f"Końcowe punkty: A={score_a}  B={score_b}  → różnica (A−B)={score_a - score_b}")
        print("Zwycięzca:", "A" if winner == game.first_player else ("B" if winner == game.second_player else "Remis"))
    return score_a - score_b, winner

#--- Statystyki + Heatmapa dla 10 ziaren losowych ---
def stats_and_heatmap(depth_min: int, depth_max: int, size: int, runs: int) -> None:
    """
    Liczy statystyki (mean, std, best, worst) dla wszystkich par głębokości A,B
    i na podstawie tych samych danych rysuje heatmapę średniej różnicy (A−B).
    """
    depths = list(range(depth_min, depth_max + 1))
    mean_matrix = np.zeros((len(depths), len(depths)))
    std_matrix = np.zeros((len(depths), len(depths)))
    best_matrix = np.zeros((len(depths), len(depths)))
    worst_matrix = np.zeros((len(depths), len(depths)))
    print(f"\n== Statystyki (mean, std, best, worst) — board {size}x{size}, runs={runs} ==")
    for i, depth_a in enumerate(depths):
        for j, depth_b in enumerate(depths):
            score_differences = []
            for seed in range(runs):  #ustawiam 10 kolejnych ziaren (0–9)
                diff, _ = play_game(size=size, depth_a=depth_a, depth_b=depth_b, seed=seed, verbose=False)
                score_differences.append(diff)
            mean_val = statistics.mean(score_differences)
            std_val = statistics.pstdev(score_differences) if len(score_differences) > 1 else 0.0
            best_val = max(score_differences)
            worst_val = min(score_differences)
            mean_matrix[i][j] = mean_val
            std_matrix[i][j] = std_val
            best_matrix[i][j] = best_val
            worst_matrix[i][j] = worst_val
            print(f"A={depth_a}, B={depth_b}: mean={mean_val:.2f}, std={std_val:.2f}, "f"best={best_val}, worst={worst_val}")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mean_matrix, cmap="coolwarm")
    ax.set_title(f"Dots & Boxes {size}×{size} — średnia różnica (A−B)")
    ax.set_xlabel("Głębokość gracza B")
    ax.set_ylabel("Głębokość gracza A")
    ax.set_xticks(range(len(depths)))
    ax.set_yticks(range(len(depths)))
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_yticklabels([str(d) for d in depths])
    #wartości w komórkach zaokrąglone
    for i in range(len(depths)):
        for j in range(len(depths)):
            ax.text(j, i, f"{mean_matrix[i, j]:.1f}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Mean score diff (A−B)")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    #pojedyncza gra
    _ = play_game(size=4, depth_a=3, depth_b=3, seed=0, verbose=True)

    #statystyki + heatmapa dla kazdej pary głębokości
    #wywoływane na 10 ziaren losowych
    stats_and_heatmap(depth_min=1, depth_max=5, size=4, runs=10)

    #przypadek trywialny
    stats_and_heatmap(depth_min=1, depth_max=1, size=4, runs=10)