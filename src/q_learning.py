from src.env import BlackjackEnv
import numpy as np
import os
import polars as pl

# Set the working directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)


# Generate relevant states (from 12 to 21 for both hard and soft totals)
states = []
for player_sum in range(12, 22):
    for dealer_card in range(1, 11):
        states.append((player_sum, dealer_card, 0))

for player_sum in range(12, 22):
    for dealer_card in range(1, 11):
        states.append((player_sum, dealer_card, 1))


### -------------------- ###
### Initialize Q-values ###

# Initialize Q-values for each state-action pair
# High values for "stand" in 20-21 and low values for "hit" in 4-11 are set to avoid suboptimal actions
q_data = {
    'State': states,
    'Action 0': np.zeros(len(states)),
    'Action 1': np.zeros(len(states))
}

for i, state in enumerate(states):
    player_sum, _, _ = state
    if player_sum >= 20:
        q_data['Action 0'][i] = +0.5
        q_data['Action 1'][i] = -0.1
    elif player_sum < 12:
        q_data['Action 0'][i] = -0.1
        q_data['Action 1'][i] = +0.5

Q = pl.DataFrame(q_data)


### ------------------------------ ###
### --- Generate count table ----- ###


# Set count ranges
count_ranges = [[0], [1, 2, 3], [4]]
betting_strategy = {
    'Bet 1': 1,
    'Bet 2': 2,
    'Bet 5': 5
}

count_data = {
    'True Count' : count_ranges,
    'Bet 1' : np.zeros(len(count_ranges)),
    'Bet 2' : np.zeros(len(count_ranges)),
    'Bet 5' : np.zeros(len(count_ranges))
}
# Initialize betting strategy for each count range (fixed in all matches)
for i, count_range in enumerate(count_ranges):
    if count_range == [0]:
        count_data['Bet 1'][i] = 0.5
        count_data['Bet 2'][i] = 0.1
        count_data['Bet 5'][i] = 0.1
    elif count_range == [1, 2, 3]:
        count_data['Bet 1'][i] = 0.2
        count_data['Bet 2'][i] = 0.5
        count_data['Bet 5'][i] = 0.2
    else:
        count_data['Bet 1'][i] = 0.2
        count_data['Bet 2'][i] = 0.3
        count_data['Bet 5'][i] = 0.5

count_df = pl.DataFrame(count_data)


# Track state-action visit counts for adaptive learning rates
visit_counts = {}
for state in states:
    visit_counts[(state, 0)] = 0
    visit_counts[(state, 1)] = 0


PARAMS = {
    "initial_lr"        : 0.1,
    "lr_decay_rate"     : 0.00005,  # Decay
    "gamma"             : 0.95,     # Discount factor
    "n_episodes"        : 200000,
    "initial_epsilon"   : 1.0,      # Start with 100% exploration
    "epsilon_min"       : 0.01,     # Minimum exploration
    "epsilon_decay"     : 0.99995,  # Decay
}


def get_state_features(full_state):
    return (full_state[0], full_state[1], full_state[2])

def get_adaptive_lr(state, action, base_lr):
    """Get state-action specific learning rate based on visit count"""
    key = (state, action)
    count = visit_counts.get(key, 0) + 1
    # Decay learning rate based on visit count, but maintain a minimum rate
    return max(base_lr / (1 + 0.005 * count), base_lr * 0.1)


def get_q_values(state_features, q_table=Q):
    """Get Q-values for a given state"""
    # Filter the DataFrame for the specific state
    state_row = q_table.filter(pl.col('State') == state_features)

    if len(state_row) == 0:
        # Return default values based on player sum
        player_sum = state_features[0]
        if player_sum < 12:
            return np.array([-0.1, 0.5])  # Default to hit for low sums
        elif player_sum >= 20:
            return np.array([0.5, -0.1])  # Default to stand for high sums
        else:
            return np.array([0.0, 0.0])  # Neutral for middle sums

    # Extract Q-values from the DataFrame
    stand_val = state_row.select('Action 0').item()
    hit_val = state_row.select('Action 1').item()
    return np.array([stand_val, hit_val])


def update_q_value(state_features, action, reward, next_state_features, lr, q_table=Q):
    """Update Q-value for state-action pair using Double Q-learning"""
    # Check if state exists in our table
    state_row = q_table.filter(pl.col('State') == state_features)
    if len(state_row) == 0:
        return  # State not in our table

    # Determine which action column to update
    action_col = 'Action 1' if action == 1 else 'Action 0'

    # Current Q-value in the DataFrame
    current_q = state_row.select(action_col).item()

    # If next_state_features is None, this is a terminal state
    if next_state_features is None:
        # Terminal state - no future rewards
        new_q = current_q + lr * (reward - current_q)
    else:
        # Get the next state's best action from current Q-table
        next_q_values = get_q_values(next_state_features, q_table)
        best_next_action = np.argmax(next_q_values)
        max_next_q = next_q_values[best_next_action]

        # Q-learning update formula with future rewards
        new_q = current_q + lr * (reward + PARAMS["gamma"] * max_next_q - current_q)

    print(f"Updating Q-value for state {state_features}, action {action}, reward {reward}, new Q-value: {new_q}")

    # Update the Q-table entry in the DataFrame
    # Create a temporary mask for the state we want to update
    mask = pl.col('State') == state_features

    # Use the when/then/otherwise pattern to update values
    q_table = q_table.with_columns(
        pl.when(mask)
        .then(pl.lit(new_q))
        .otherwise(pl.col(action_col))
        .alias(action_col)
    )
    print(f"Updated Q-table for state {state_features}: {q_table.filter(mask)}")

    # Track visit counts
    visit_counts[(state_features, action)] = visit_counts.get((state_features, action), 0) + 1

    return q_table


def update_count_table(current_count, reward, count_df=count_df):
    """Update the value associated with the move of betting a certain amount based on the reward received"""
    # Check if reward is positive or negative
    if reward > 0:
        # Positive reward - increase the value of betting a higher amount and decrease the value of betting a lower amount for the row corresponding to the current count
        # Find the index of the current count range
        index = None
        for i, count_range in enumerate(count_ranges):
            if current_count in count_range:
                index = i
                break
        # update the row corresponding to the current count
        if index is None:
            raise ValueError("Current count not found in count ranges")
        # Update the values in the count_df DataFrame
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 1'][index] * 0.9))
            .otherwise(pl.col('Bet 1'))
            .alias('Bet 1')
        )
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 2'][index] * 0.95))
            .otherwise(pl.col('Bet 2'))
            .alias('Bet 2')
        )
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 5'][index] * 1.05))
            .otherwise(pl.col('Bet 5'))
            .alias('Bet 5')
        )
    elif reward < 0:
        # Negative reward - decrease the value of betting a higher amount and increase the value of betting a lower amount for the row corresponding to the current count
        # Find the index of the current count range
        index = None
        for i, count_range in enumerate(count_ranges):
            if current_count in count_range:
                index = i
                break
        # update the row corresponding to the current count
        if index is None:
            raise ValueError("Current count not found in count ranges")
        # Update the values in the count_df DataFrame
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 1'][index] * 1.05))
            .otherwise(pl.col('Bet 1'))
            .alias('Bet 1')
        )
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 2'][index] * 0.95))
            .otherwise(pl.col('Bet 2'))
            .alias('Bet 2')
        )
        count_df = count_df.with_columns(
            pl.when(pl.col('True Count') == count_ranges[index])
            .then(pl.lit(count_df['Bet 5'][index] * 0.9))
            .otherwise(pl.col('Bet 5'))
            .alias('Bet 5')
        )
    else:
        # do nothing
        # ALTERNATIVE: update something
        pass

    return count_df


config = {
    "num_decks": 6,
    "red_card_position": 0.2,
    "bet_size": [1],
    "actions": ["stand", "hit"],
    "num_players": 1
}
# Create environment with 6 decks (standard casino configuration)
env = BlackjackEnv(config=config)

# Training loop with convergence check
print("Starting improved training...")
wins = 0
draws = 0
losses = 0
epsilon = PARAMS["initial_epsilon"]
lr = PARAMS["initial_lr"]
money_won = 0
money_lost = 0

# Parameters for convergence
n_episodes = 100  # Number of episodes for training
convergence_threshold = 0.001  # Lower threshold for better stability
convergence_check_interval = 10000  # Check for convergence every N episodes
convergence_required_count = 3  # Number of consecutive checks below threshold to confirm convergence
max_episodes = n_episodes  # Maximum episodes as a fallback

# Keep a copy of the previous Q-table for comparison
previous_q = Q.clone()
convergence_count = 0
converged = False
episode = 0
# first training phase only for the Q-table with fixed betting strategy
while episode < max_episodes and not converged:

    env.reset()
    bet_index = env.bet_space.sample()  # Sample bet index from the environment
    bet_amount = env.bets[bet_index]  # Sample bet amount from the environment
    state, reward, done = env.step(bet_index, action_type="bet")  # Place bet
    # print(bet_amount)
    state_features = get_state_features(state)

    # Training episode
    while not done:

        if state_features[0] < 12:
            # Always hit this state as it's not relevant for our training
            next_state, _, _ = env.step(1, action_type="move")
            next_state_features = get_state_features(next_state) if not done else None
            state = next_state
            state_features = next_state_features if next_state is not None else None
            continue

        # Epsilon-greedy action selection
        elif np.random.rand() < epsilon:
            action = env.move_space.sample()  # Random action
        else:
            q_values = get_q_values(state_features)
            action = np.argmax(q_values)  # Greedy action

        # Take action
        next_state, reward, done = env.step(action, action_type="move")
        next_state_features = get_state_features(next_state) if not done else None

        # Get adaptive learning rate for this state-action pair
        adaptive_lr = get_adaptive_lr(state_features, action, lr)

        # Randomly decide which Q-table to update (Double Q-learning)
        # print(f"State: {state_features}, Action: {action}, Done: {done}, Reward: {reward}, Next State: {next_state_features}")
        Q = update_q_value(state_features, action, reward * bet_amount, next_state_features, adaptive_lr, Q)

        # Track outcomes
        if done:
            if reward > 0:
                wins += 1
                money_won += reward * bet_amount
            elif reward == 0:
                draws += 1
            else:
                losses += 1
                money_lost += abs(reward) * bet_amount

        state = next_state
        state_features = next_state_features if next_state is not None else None

        if state_features is None:
            break

    # Decay epsilon and learning rate
    epsilon = max(PARAMS["epsilon_min"], epsilon * PARAMS["epsilon_decay"])
    lr = PARAMS["initial_lr"] / (1 + PARAMS["lr_decay_rate"] * episode)

    # Check for convergence periodically
    if episode % convergence_check_interval == 0 and episode > 0:
        # Calculate the maximum absolute difference between current and previous Q-values
        diff_stand = (Q.select('Action 0').to_numpy() -
                      previous_q.select('Action 0').to_numpy())
        diff_hit = (Q.select('Action 1').to_numpy() -
                    previous_q.select('Action 1').to_numpy())

        max_diff_stand = np.max(np.abs(diff_stand))
        max_diff_hit = np.max(np.abs(diff_hit))
        max_diff = max(max_diff_stand, max_diff_hit)

        if max_diff < convergence_threshold:
            convergence_count += 1
            print(
                f"Episode {episode}, max Q-value change: {max_diff:.6f} (convergence count: {convergence_count}/{convergence_required_count})")
            if convergence_count >= convergence_required_count:
                print(f"Converged after {episode} episodes (max Q-value change: {max_diff:.6f})")
                converged = True
        else:
            convergence_count = 0
            print(f"Episode {episode}, max Q-value change: {max_diff:.6f}")

        # Store current Q-values for next comparison
        previous_q = Q.clone()

    episode += 1

# Final statistics
total_episodes = episode
print(f"Training complete after {total_episodes} episodes.")
print(f"Win rate: {wins / total_episodes:.4f}")
print(f"Draw rate: {draws / total_episodes:.4f}")
print(f"Loss rate: {losses / total_episodes:.4f}")