import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def extract_state_info(state):

    parts = state.strip('"()').split(",")
    if len(parts) != 3:
        raise ValueError("Invalid state format")
    player = int(parts[0].strip())
    dealer = int(parts[1].strip())
    soft_hand = int(parts[2].strip())

    return player, dealer, soft_hand


def parse_strategy(file_path):

    strategy = pd.read_csv(file_path)
    policy = {}
    for _, row in strategy.iterrows():
        player, dealer, soft_hand = extract_state_info(row["State"])
        policy[(player, dealer, soft_hand)] = row["Action"]
    
    player_hand = sorted(set([k[0] for k in policy.keys()]))
    dealer_hand = sorted(set([k[1] for k in policy.keys()]))
    
    matrix_hard = np.ones((len(player_hand), len(dealer_hand)), dtype=int)
    matrix_soft = np.ones((len(player_hand), len(dealer_hand)), dtype=int)
    
    player_map = {val: i for i, val in enumerate(player_hand)}
    dealer_map = {val: i for i, val in enumerate(dealer_hand)}
    
    for (player, dealer, soft), action in policy.items():
        if player in player_map and dealer in dealer_map:
            if soft == 0:
                matrix_hard[player_map[player], dealer_map[dealer]] = action
            else:
                matrix_soft[player_map[player], dealer_map[dealer]] = action
    
    return matrix_hard, matrix_soft, player_hand, dealer_hand


def draw_matrix(matrix, player_hand, dealer_hand, title, ax):

    moves_matrix = np.where(matrix == 0, "Stand", "Hit")
    
    sns.heatmap(
        matrix,
        annot=moves_matrix,
        fmt="", 
        cmap=["red", "green"],
        cbar=False,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title)
    ax.set_xlabel("Dealer Hand")
    ax.set_ylabel("Player Hand")
    
    # Set tick labels
    ax.set_xticks(np.arange(len(dealer_hand)) + 0.5)
    ax.set_yticks(np.arange(len(player_hand)) + 0.5)
    ax.set_xticklabels(dealer_hand)
    ax.set_yticklabels(player_hand)


def plot_policy(file_path):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Blackjack Policy")

    matrix_hard, matrix_soft, player_hand, dealer_hand = parse_strategy(file_path)
    draw_matrix(matrix_hard, player_hand, dealer_hand, "Hard Hands", axes[0])
    draw_matrix(matrix_soft, player_hand, dealer_hand, "Soft Hands", axes[1])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    file_path = "basic_strategy.csv"
    plot_policy(file_path)
