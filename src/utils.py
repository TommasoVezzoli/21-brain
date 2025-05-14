import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np



def parse_strategy_csv(file_path):
    """"    
    Parse a CSV file containing a strategy table.
    Returns a dictionary with state keys and Q values.
    """
    try:
        # Read the CSV file
        df = pl.read_csv(file_path)
        
        # Initialize the basic strategy dictionary
        strategy = {}
        
        # Process the dataframe into a dictionary
        for row in df.iter_rows(named=True):
            # Parse the state from string format like '(12, 10, 0)'
            # Extract the state values
            state_str = row['State'].strip('()').split(', ')

            # state_str = row['State'].strip('[]').split()
            player_sum = int(state_str[0])
            dealer_card = int(state_str[1])
            usable_ace = int(state_str[2])
            
            # Create the state key
            state_key = (player_sum, dealer_card, usable_ace)
            
            # Get the action values
            stand_value = row['Action 0 (Stand)']
            hit_value = row['Action 1 (Hit)']
            # Check if the column for double action exists
            if 'Action 2 (Double)' in row:
                double_value = row['Action 2 (Double)']
            else:
                # If not present, set double value to None or some default
                double_value = None
            
            # Store the action values in a dictionary
            strategy[state_key] = {
                0: stand_value,  # Stand
                1: hit_value,    # Hit
                2: double_value   # Double
            }
            
        return strategy
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
    



# Function to parse betting strategy
def parse_betting_strategy_csv(file_path):
    """
    Parse a CSV file containing a betting strategy table.
    Returns a dictionary with true count keys and bet values.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Initialize the betting strategy dictionary
        betting_strategy = {}
        
        # Process the dataframe into a dictionary
        for _, row in df.iterrows():
            true_count = int(row['true_count'])
            
            # Get the bet values for each true count
            bet_values = {}
            for bet in [1, 4, 8]:  # Assuming these are the possible bet sizes
                if f"{bet}" in df.columns:
                    bet_values[bet] = row[f"{bet}"]
            
            betting_strategy[true_count] = bet_values
            
        return betting_strategy
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
    



def parse_split_strategy_csv(file_path):
    """
    Parse a CSV file containing a split strategy table.
    Returns a dictionary with state keys and split values.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Initialize the split strategy dictionary
        split_strategy = {}
        
        # Process the dataframe into a dictionary
        for _, row in df.iterrows():
            # Extract the state key
            pair_value = int(row['Pair Value'])
            dealer_upcard = int(row['Dealer Upcard'])
            state_key = (pair_value, dealer_upcard)
            
            # Get the split and no-split values
            split_value = row['Split']
            no_split_value = row['No Split']
            
            # Store the values in the dictionary
            split_strategy[state_key] = {
                0: no_split_value,  # No Split
                1: split_value      # Split
            }
            
        return split_strategy
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}
    



def get_state_features(observation):
    """Extract meaningful features from the observation for basic strategy decisions"""
    player_sum = observation["player_score"]
    dealer_card = observation["dealer_upcard"]
    usable_ace = observation["soft_hand"]
    return (player_sum, dealer_card, usable_ace)



def get_true_count(observation):
    """Extract the true count from the observation"""
    true_count = observation["true_count"]
    return true_count



def discretize_true_count(true_count):
    """Discretize the true count into bins for Q-learning"""
    if true_count <= 1: return 0
    elif 1 < true_count < 5: return 1
    else: return 2



def get_best_bet(true_count, betting_strat):
    """Get the best bet size based on the true count"""
    # Get the bet values for the given true count
    bet_values = betting_strat.get(true_count, {})
    # Find the bet amount with the highest value
    best_bet = max(bet_values, key=bet_values.get, default=None)
    return best_bet



def get_split_state(observation):
    """Extract the pair value and dealer upcard for split decisions"""
    pair_value = observation["pair_value"]
    dealer_upcard = observation["dealer_upcard"]
    return (pair_value, dealer_upcard)



def get_adaptive_lr(state, action, base_lr, visit_counts):
    """Get state-action specific learning rate based on visit count"""
    key = (state, action)
    count = visit_counts.get(key, 0) + 1
    # Decay learning rate based on visit count, but maintain a minimum rate
    return max(base_lr / (1 + 0.005 * count), base_lr * 0.1)


def get_adaptive_lr_v2(initial_lr, min_lr, decay_rate, episode, visits=None):
    """Get state-action specific learning rate based on visit count (version 2 used for betting and splitting)"""
    if visits and visits > 100:
        # Slower decay for frequently visited states
        return max(min_lr, initial_lr * (0.9995 ** (visits // 100)))
    else:
        # Regular decay based on episode number
        return max(min_lr, initial_lr * (decay_rate ** episode))



def visualize_hit_stand(basic_strat_file, q_table_df):
    """
    Visualize and compare blackjack policies from two CSV files.
    
    Parameters:
    basic_strat_file (str): Path to the basic strategy CSV file
    q_table_file (str): Path to the Q-table CSV file
    """
    # Read the CSV files
    basic_strat_df = pd.read_csv(basic_strat_file)
    q_table_df = pd.read_csv(q_table_df)

    def extract_state_info(state_str):
        # Extract values from format like "(12, 3, 0)"
        state_str = state_str.strip('()').split(',')
        player_value = int(state_str[0])
        dealer_card = int(state_str[1])
        usable_ace = int(state_str[2])
        return player_value, dealer_card, usable_ace
    
    # Create mapping dictionaries for both dataframes
    basic_policy = {}
    q_policy = {}
    
    # Process basic strategy data
    for _, row in basic_strat_df.iterrows():
        try:
            player_value, dealer_card, usable_ace = extract_state_info(row['State'])
            basic_policy[(player_value, dealer_card, usable_ace)] = row['Best Action']
        except:
            print(f"Couldn't parse state: {row['State']}")
            
    # Process Q-table data
    for _, row in q_table_df.iterrows():
        try:
            player_value, dealer_card, usable_ace = extract_state_info(row['State'])
            q_policy[(player_value, dealer_card, usable_ace)] = row['Best Action']
        except:
            print(f"Couldn't parse state: {row['State']} in double q")
    
    # Prepare data for visualization
    # Define the range of player values and dealer cards
    player_values = sorted(set([k[0] for k in basic_policy.keys()]))
    dealer_cards = sorted(set([k[1] for k in basic_policy.keys()]))
    
    # Create matrices for visualization (one for non-usable ace, one for usable ace)
    # 0 = no usable ace, 1 = usable ace
    basic_matrix_no_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    basic_matrix_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    q_matrix_no_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    q_matrix_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    
    # Map player values and dealer cards to indices
    player_map = {val: i for i, val in enumerate(player_values)}
    dealer_map = {val: i for i, val in enumerate(dealer_cards)}
    
    # Fill the matrices
    for (player, dealer, ace), action in basic_policy.items():
        if player in player_map and dealer in dealer_map:
            if ace == 0:
                basic_matrix_no_ace[player_map[player], dealer_map[dealer]] = action
            else:
                basic_matrix_ace[player_map[player], dealer_map[dealer]] = action
    
    for (player, dealer, ace), action in q_policy.items():
        if player in player_map and dealer in dealer_map:
            if ace == 0:
                q_matrix_no_ace[player_map[player], dealer_map[dealer]] = action
            else:
                q_matrix_ace[player_map[player], dealer_map[dealer]] = action
    
    # Replace empty cells with "Hit"
    basic_matrix_no_ace = np.where(basic_matrix_no_ace == 0, "Hit", basic_matrix_no_ace)
    basic_matrix_ace = np.where(basic_matrix_ace == 0, "Hit", basic_matrix_ace)
    q_matrix_no_ace = np.where(q_matrix_no_ace == 0, "Hit", q_matrix_no_ace)
    q_matrix_ace = np.where(q_matrix_ace == 0, "Hit", q_matrix_ace)

    basic_matrix_no_ace = basic_matrix_no_ace[4:,:]
    basic_matrix_ace = basic_matrix_ace[4:,:]
    q_matrix_no_ace = q_matrix_no_ace[4:,:]
    q_matrix_ace = q_matrix_ace[4:,:]
    
    # Visualization function
    def plot_policy_matrix(matrix, title, ax):
        # Convert 'Hit'/'Stand' to numeric for coloring
        numeric_matrix = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                numeric_matrix[i, j] = 1 if matrix[i, j] == 'Stand' else 0
        
        # Create heatmap
        sns.heatmap(
            numeric_matrix, 
            annot=matrix, 
            fmt='', 
            cmap=['red','limegreen'],
            cbar=False,
            linewidths=0.5,
            ax=ax
        )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Player Value')
        
        # Set tick labels
        ax.set_xticks(np.arange(len(dealer_cards)) + 0.5)
        ax.set_yticks(np.arange(len(player_values[4:])) + 0.5)
        ax.set_xticklabels(dealer_cards)
        ax.set_yticklabels(player_values[4:])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # fig, axes = plt.subplots(1, 2, figsize=(15, 10))

    # plot_policy_matrix(basic_matrix_no_ace, "Basic Strategy - No Usable Ace", axes[ 0])
    # plot_policy_matrix(basic_matrix_ace, "Basic Strategy - Usable Ace", axes[1])
    
    plot_policy_matrix(basic_matrix_no_ace, "Basic Strategy - No Usable Ace", axes[0, 0])
    plot_policy_matrix(basic_matrix_ace, "Basic Strategy - Usable Ace", axes[0, 1])
    plot_policy_matrix(q_matrix_no_ace, "Q-Learning Strategy - No Usable Ace", axes[1, 0])
    plot_policy_matrix(q_matrix_ace, "Q-Learning Strategy - Usable Ace", axes[1, 1])
    
    plt.tight_layout()
    # plt.savefig("blackjack_policy_comparison_new.png", dpi=300, bbox_inches='tight')
    plt.show()




def visualize_hit_stand_dd(basic_strat_file, q_table_file):
    """
    Visualize and compare blackjack policies from two CSV files with strategies for hit/stand/double down.
    
    Parameters:
    basic_strat_file (str): Path to the basic strategy CSV file
    q_table_file (str): Path to the Q-table CSV file
    """
    # Read the CSV files
    basic_strat_df = pd.read_csv(basic_strat_file)
    q_table_file = pd.read_csv(q_table_file)

    def extract_state_info(state_str):
        # Extract values from format like "(12, 3, 0)"
        state_str = state_str.strip('()').split(',')
        player_value = int(state_str[0])
        dealer_card = int(state_str[1])
        usable_ace = int(state_str[2])
        return player_value, dealer_card, usable_ace
    
    # Create mapping dictionaries for both dataframes
    basic_policy = {}
    q_policy = {}
    
    # Process basic strategy data
    for _, row in basic_strat_df.iterrows():
        try:
            player_value, dealer_card, usable_ace = extract_state_info(row['State'])
            basic_policy[(player_value, dealer_card, usable_ace)] = row['Best Action']
        except:
            print(f"Couldn't parse state: {row['State']}")
            
    # Process Q-table data
    for _, row in q_table_file.iterrows():
        try:
            player_value, dealer_card, usable_ace = extract_state_info(row['State'])
            q_policy[(player_value, dealer_card, usable_ace)] = row['Best Action']
        except:
            print(f"Couldn't parse state: {row['State']} in double q")
    
    # Prepare data for visualization
    # Define the range of player values and dealer cards
    player_values = sorted(set([k[0] for k in basic_policy.keys()]))
    dealer_cards = sorted(set([k[1] for k in basic_policy.keys()]))
    
    # Create matrices for visualization (one for non-usable ace, one for usable ace)
    # 0 = no usable ace, 1 = usable ace
    basic_matrix_no_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    basic_matrix_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    q_matrix_no_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    q_matrix_ace = np.zeros((len(player_values), len(dealer_cards)), dtype=object)
    
    # Map player values and dealer cards to indices
    player_map = {val: i for i, val in enumerate(player_values)}
    dealer_map = {val: i for i, val in enumerate(dealer_cards)}
    
    # Fill the matrices
    for (player, dealer, ace), action in basic_policy.items():
        if player in player_map and dealer in dealer_map:
            if ace == 0:
                basic_matrix_no_ace[player_map[player], dealer_map[dealer]] = action
            else:
                basic_matrix_ace[player_map[player], dealer_map[dealer]] = action
    
    for (player, dealer, ace), action in q_policy.items():
        if player in player_map and dealer in dealer_map:
            if ace == 0:
                q_matrix_no_ace[player_map[player], dealer_map[dealer]] = action
            else:
                q_matrix_ace[player_map[player], dealer_map[dealer]] = action
    
    # Set default value for missing states in Q-table with usable_ace = 1
    for i in range(len(player_values)):
        for j in range(len(dealer_cards)):
            if q_matrix_ace[i, j] == 0:  # Default value for missing states
                q_matrix_ace[i, j] = 'Hit'
            if basic_matrix_ace[i, j] == 0:
                basic_matrix_ace[i, j] = 'Hit'
    
    basic_matrix_no_ace = basic_matrix_no_ace[4:,:]
    basic_matrix_ace = basic_matrix_ace[4:,:]
    q_matrix_no_ace = q_matrix_no_ace[4:,:]
    q_matrix_ace = q_matrix_ace[4:,:]
    
    # Visualization function
    def plot_policy_matrix(matrix, title, ax):
        # Convert 'Hit'/'Stand'/'Double' to numeric for coloring
        numeric_matrix = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i, j] == 'Stand':
                    numeric_matrix[i, j] = 1
                elif matrix[i, j] == 'Hit':
                    numeric_matrix[i, j] = 0
                elif matrix[i, j] == 'Double':
                    numeric_matrix[i, j] = 2
        # Create heatmap
        # sns.heatmap(numeric_matrix, annot=matrix, fmt='', cmap='gist_rainbow', cbar=False, ax=ax,
        #             xticklabels=dealer_cards, yticklabels=player_values[4:], linewidths=.5)
        sns.heatmap(
            numeric_matrix, 
            annot=matrix, 
            fmt='', 
            cmap=['red','limegreen', 'pink'],
            cbar=False,
            linewidths=0.5,
            ax=ax
        )
        
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Player Value')
        
        # Set tick labels
        ax.set_xticks(np.arange(len(dealer_cards)) + 0.5)
        ax.set_yticks(np.arange(len(player_values[4:])) + 0.5)
        ax.set_xticklabels(dealer_cards)
        ax.set_yticklabels(player_values[4:])
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_policy_matrix(basic_matrix_no_ace, "Basic Strategy - No Usable Ace", axes[0, 0])
    plot_policy_matrix(basic_matrix_ace, "Basic Strategy - Usable Ace", axes[0, 1])
    plot_policy_matrix(q_matrix_no_ace, "Q-Learning Strategy - No Usable Ace", axes[1, 0])
    plot_policy_matrix(q_matrix_ace, "Q-Learning Strategy - Usable Ace", axes[1, 1])
    
    plt.tight_layout()
    plt.show()




def visualize_true_count(q_table_file):
    # Load the Q-table CSV file with average Q-values for each bet size and true count
    bet_avg = pd.read_csv(q_table_file)

    # Melt the DataFrame to long format for seaborn
    bet_avg_long = bet_avg.melt(id_vars="true_count", var_name="bet_size", value_name="q_value")
    bet_avg_long["bet_size"] = bet_avg_long["bet_size"].astype(str)  # Ensure bet sizes are treated as categories

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bet_avg_long, x="true_count", y="q_value", hue="bet_size", marker="o")
    plt.title("Average Q-values of Bet Sizes vs. True Count")
    plt.xlabel("True Count Bucket")
    plt.ylabel("Average Q-value")
    plt.legend(title="Bet Size")
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def visualize_split(basic_strat_file, q_table_file):
    """
    Visualize and compare blackjack policies from two CSV files with strategies for split decisions.
    
    Parameters:
    basic_strat_file (str): Path to the basic strategy CSV file
    q_table_file (str): Path to the Q-table CSV file
    """
    # Read the CSV files
    basic_strat_df = pd.read_csv(basic_strat_file)
    q_table_df = pd.read_csv(q_table_file)

    # Create mapping dictionaries for both dataframes
    basic_policy = {}
    q_policy = {}

    # Process basic strategy data
    for _, row in basic_strat_df.iterrows():
        try:
            pair_value = int(row['Pair Value'])
            dealer_card = int(row['Dealer Upcard'])
            basic_policy[(pair_value, dealer_card)] = row['Action']
        except Exception as e:
            print(f"Couldn't parse row in basic strategy: {row}, error: {e}")

    # Process Q-table data
    for _, row in q_table_df.iterrows():
        try:
            pair_value = int(row['Pair Value'])
            dealer_card = int(row['Dealer Upcard'])
            q_policy[(pair_value, dealer_card)] = row['Action']
        except Exception as e:
            print(f"Couldn't parse row in Q-table: {row}, error: {e}")

    # Define the range of pair values and dealer cards
    pair_values = sorted(set([k[0] for k in basic_policy.keys()]))
    dealer_cards = sorted(set([k[1] for k in basic_policy.keys()]))

    # Create matrices for visualization
    basic_matrix = np.zeros((len(pair_values), len(dealer_cards)), dtype=object)
    q_matrix = np.zeros((len(pair_values), len(dealer_cards)), dtype=object)

    # Map pair values and dealer cards to indices
    pair_map = {val: i for i, val in enumerate(pair_values)}
    dealer_map = {val: i for i, val in enumerate(dealer_cards)}

    # Fill the matrices
    for (pair, dealer), action in basic_policy.items():
        if pair in pair_map and dealer in dealer_map:
            basic_matrix[pair_map[pair], dealer_map[dealer]] = action

    for (pair, dealer), action in q_policy.items():
        if pair in pair_map and dealer in dealer_map:
            q_matrix[pair_map[pair], dealer_map[dealer]] = action

    # Replace empty cells with "No Split"
    basic_matrix = np.where(basic_matrix == 0, "No Split", basic_matrix)
    q_matrix = np.where(q_matrix == 0, "No Split", q_matrix)

    # Visualization function
    def plot_policy_matrix(matrix, title, ax):
        # Convert 'Split'/'No Split' to numeric for coloring
        numeric_matrix = np.zeros(matrix.shape)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                numeric_matrix[i, j] = 1 if matrix[i, j] == 'Split' else 0

        # Create heatmap
        sns.heatmap(
            numeric_matrix,
            annot=matrix,
            fmt='',
            cmap=['yellow', 'lightblue'],
            cbar=False,
            linewidths=0.5,
            ax=ax
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel('Dealer Card')
        ax.set_ylabel('Pair Value')

        # Set tick labels
        ax.set_xticks(np.arange(len(dealer_cards)) + 0.5)
        ax.set_yticks(np.arange(len(pair_values)) + 0.5)
        ax.set_xticklabels(dealer_cards)
        ax.set_yticklabels(pair_values)

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    plot_policy_matrix(basic_matrix, "Basic Strategy - Split Decisions", axes[0])
    plot_policy_matrix(q_matrix, "Q-Learning Strategy - Split Decisions", axes[1])

    plt.tight_layout()
    plt.show()