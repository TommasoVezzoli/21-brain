from blackjack import *
import gymnasium as gym


class BlackjackEnv(gym.Env):
    """
    Custom Blackjack environment from OpenAI Gym.
    Supports the following options:
        - num_decks: Number of decks in the shoe (default 6)
        - cut_card_fraction: Fraction of the shoe at which to reshuffle (default 0.2)
        - bets: Sizes of allowed bets (default [1])
        - actions: List of valid actions (default ["hit", "stand"])
        - num_players: Number of players at the table (default 1)
        - max_splits: Maximum number of splits allowed (default 3)
        - split_aces_only_one_card: If True, split aces can only receive one card (default True)

    :param config: Configuration dictionary
    """

    def __init__(self, config):

        # Read the configuration file and set the parameters
        self.num_decks = config.get("num_decks", 6)
        self.cut_card_position = config.get("red_card_position", 0.2)
        self.bets = config.get("bet_size", [1])
        self.actions = config.get("actions", ["stand", "hit", "double", "split"])
        self.num_players = config.get("num_players", 1)
        self.max_splits = config.get("max_splits", 3)
        self.split_aces_only_one_card = config.get("split_aces_only_one_card", True)

        # Configure the action and observation spaces
        self.bet_space = gym.spaces.Discrete(len(self.bets))
        self.move_space = gym.spaces.Discrete(len(self.actions))

        # Updated observation space to include whether the hand can be split
        self.observation_space = gym.spaces.Dict({
            "player_score": gym.spaces.Discrete(32),      # Player's current hand score
            "dealer_upcard": gym.spaces.Discrete(11),     # Dealer's upcard value
            "soft_hand": gym.spaces.Discrete(2),          # Whether the hand is soft
            "can_split": gym.spaces.Discrete(2),          # Whether the hand can be split
            "pair_value": gym.spaces.Discrete(11),        # Value of the pair (if any)
            "true_count": gym.spaces.Box(                 # Count
                low=-10,
                high=10,
                shape=(1,),
                dtype=float
            )
        })

        self.table = Table(
            deck=Deck(num_decks=self.num_decks, cut_card_position=self.cut_card_position),
            num_players=self.num_players
        )
        self.table.max_splits = self.max_splits

        self.episode_reward = 0
        self.last_split = False
    
    def _get_observation(self):
        """Get the current observation state"""
        player = self.table.players[0]
        current_hand = player.get_current_hand()
        
        if current_hand is None or len(current_hand.cards) == 0:
            # Initial state or no current hand
            return {
                "player_score": 0,
                "dealer_upcard": 0,
                "soft_hand": 0,
                "can_split": 0,
                "pair_value": 0,
                "true_count": self.table.counter.true_count
            }
        
        # Get dealer's upcard value
        dealer_val = 0
        if len(self.table.dealer.hands[0].cards) > 0:
            dealer_val = self.table.dealer.hands[0].cards[0].val
            dealer_val = 11 if dealer_val == 1 else dealer_val  # Convert Ace to 11
        
        # Check if the hand can be split
        can_split = 1 if self.table.can_player_split(0) and current_hand.can_split() else 0
        
        # Get the pair value if the hand can be split
        pair_value = 0
        if can_split:
            pair_value = current_hand.cards[0].val
            if pair_value == 1:  # Convert Ace to 11 for observation
                pair_value = 11
        
        return {
            "player_score": current_hand.get_value(),
            "dealer_upcard": dealer_val,
            "soft_hand": int(current_hand.check_soft()),
            "can_split": can_split,
            "pair_value": pair_value,
            "true_count": self.table.counter.true_count
        }

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Initialize the environment state
        if self.table.deck.check_cut_card():
            self.table.deck.reset()
            self.table.reset_counter()
        self.table.reset()

        self.episode_reward = 0
        self.last_split = False

        observation = self._get_observation()
        return observation

    def step(self, action, action_type="move"):
        reward = 0
        done = False
        info = {}

        if action_type == "bet":
            bet = self.bets[action]
            self.table.players[0].hands[0].bet = bet

            # Deal the cards and handle potential naturals
            self.table.deal()

            p_natural = self.table.players[0].hands[0].check_natural()
            d_natural = self.table.dealer.hands[0].check_natural()
            if p_natural or d_natural:
                if p_natural and d_natural:
                    reward = 0
                elif p_natural:
                    reward = 1.5
                else:
                    reward = -1
                done = True
                self.episode_reward += reward
            
            observation = self._get_observation()
            return observation, reward, done, info
        
        elif action_type == "move":
            player = self.table.players[0]
            current_hand = player.get_current_hand()
            
            if current_hand is None:
                # No more hands to play, end the round
                reward = self.episode_reward
                done = True
                observation = self._get_observation()
                return observation, reward, done, info
            
            action_str = self.actions[action]
            
            # Process the action
            if action_str == "hit":
                self.table.deal_card(player=0)
                current_hand = player.get_current_hand()  # Get current hand again after dealing

                # Check if the hand is busted or reached 21
                if current_hand.check_bust():
                    reward = -1
                    current_hand.is_done = True
                    self.episode_reward += reward
                    
                    # # Move to the next hand if there is one
                    # if not player.move_to_next_hand():
                    #     done = True
                    #     # If no more hands, we've played through all hands
                    #     if player.current_hand_idx >= len(player.hands):
                    #         done = True
                    #     else:
                    #         # Deal a card to the new hand if it doesn't have cards yet
                    #         next_hand = player.get_current_hand()
                    #         if next_hand and len(next_hand.cards) == 1:  # Split hand with only one card
                    #             self.table.deal_card(player=0)
                    # Move to the next hand if there is one
                    if not player.move_to_next_hand():
                        done = True
                    else:
                        # Deal a card to the new hand if it doesn't have cards yet
                        next_hand = player.get_current_hand()
                        if next_hand and len(next_hand.cards) == 1:  # Split hand with only one card
                            self.table.deal_card(player=player)  # Use dynamic player reference
                
                elif current_hand.get_value() == 21:
                    # Automatically stand on 21
                    current_hand.is_done = True
                    
                    # Move to the next hand if there is one
                    if not player.move_to_next_hand():
                        # Play the dealer and score all hands
                        while self.table.dealer.move() == "hit":
                            self.table.deal_card("dealer")
                        
                        # Score all completed hands against the dealer
                        dealer_bust = self.table.dealer.hands[0].check_bust()
                        dealer_score = self.table.dealer.hands[0].get_score()
                        
                        for hand in player.hands:
                            if not hand.check_bust():
                                if dealer_bust:
                                    reward += 1
                                else:
                                    player_score = hand.get_score()
                                    if player_score > dealer_score:
                                        reward += 1
                                    elif player_score < dealer_score:
                                        reward -= 1
                        
                        self.episode_reward += reward
                        done = True
                    else:
                        # Deal a card to the new hand if it doesn't have cards yet
                        next_hand = player.get_current_hand()
                        if next_hand and len(next_hand.cards) == 1:  # Split hand with only one card
                            self.table.deal_card(player=0)
                
            elif action_str == "stand":
                current_hand.is_done = True
                
                # If this is the last hand or the only hand, play the dealer's hand
                if not player.move_to_next_hand():
                    # Play dealer hand only for the last player hand
                    while self.table.dealer.move() == "hit":
                        self.table.deal_card("dealer")
                    
                    # Score all hands against the dealer
                    dealer_bust = self.table.dealer.hands[0].check_bust()
                    dealer_score = self.table.dealer.hands[0].get_score()
                    
                    for hand in player.hands:
                        if not hand.check_bust():
                            if dealer_bust:
                                reward += 1
                            else:
                                player_score = hand.get_score()
                                if player_score > dealer_score:
                                    reward += 1
                                elif player_score < dealer_score:
                                    reward -= 1
                    
                    self.episode_reward += reward
                    done = True
                else:
                    # Deal a card to the new hand if it doesn't have cards yet
                    next_hand = player.get_current_hand()
                    if next_hand and len(next_hand.cards) == 1:  # Split hand with only one card
                        self.table.deal_card(player=0)
                    
            elif action_str == "double":
                # Double the bet and take exactly one more card
                self.table.deal_card(player=0)
                current_hand.is_done = True
                
                if current_hand.check_bust():
                    reward = -1*2
                    self.episode_reward += reward #possibly multiply by 2
                
                # Move to the next hand if there is one
                if not player.move_to_next_hand():
                    # If no more hands, play the dealer and score all hands
                    while self.table.dealer.move() == "hit":
                        self.table.deal_card("dealer")
                    
                    dealer_bust = self.table.dealer.hands[0].check_bust()
                    dealer_score = self.table.dealer.hands[0].get_score()
                    
                    for hand in player.hands:
                        if not hand.check_bust():
                            if dealer_bust:
                                reward += 1*2
                            else:
                                player_score = hand.get_score()
                                if player_score > dealer_score:
                                    reward += 1*2
                                elif player_score < dealer_score:
                                    reward -= 1*2
                    
                    self.episode_reward += reward #possibly multiply by 2
                    done = True
                else:
                    # Deal a card to the new hand if it doesn't have cards yet
                    next_hand = player.get_current_hand()
                    if next_hand and len(next_hand.cards) == 1:  # Split hand with only one card
                        self.table.deal_card(player=0)
            
            elif action_str == "split" and self.table.can_player_split(0):
                # Split the current hand
                current_hand = player.get_current_hand()
                
                if current_hand.can_split():
                    # Split the hand
                    new_hand = current_hand.split()
                    player.hands.insert(player.current_hand_idx + 1, new_hand)
                    
                    # Deal one card to the first hand
                    self.table.deal_card(0)
                    
                    # Special handling for split aces
                    if current_hand.cards[0].rank == "A" and self.split_aces_only_one_card:
                        current_hand.is_done = True
                        
                        # Move to the second hand, deal a card, and mark it done too
                        player.move_to_next_hand()
                        self.table.deal_card(0)
                        player.get_current_hand().is_done = True
                        
                        # Move to next hand if any, otherwise resolve the round
                        if not player.move_to_next_hand():
                            # Play dealer's hand
                            while self.table.dealer.move() == "hit":
                                self.table.deal_card("dealer")
                            
                            # Score all hands
                            dealer_bust = self.table.dealer.hands[0].check_bust()
                            dealer_score = self.table.dealer.hands[0].get_score()
                            
                            for hand in player.hands:
                                if not hand.check_bust():
                                    if dealer_bust:
                                        reward += 1
                                    else:
                                        player_score = hand.get_score()
                                        if player_score > dealer_score:
                                            reward += 1
                                        elif player_score < dealer_score:
                                            reward -= 1
                            
                            self.episode_reward += reward
                            done = True
                    
                    # For non-ace splits, continue with the first hand
                    # The second hand will receive its card when we move to it
                    player.move_to_next_hand()
                    self.table.deal_card(0) 
                    player.move_to_previous_hand()

                else:
                    # Invalid split attempt - treat as stand
                    current_hand.is_done = True
                    
                    if not player.move_to_next_hand():
                        # Play dealer's hand
                        while self.table.dealer.move() == "hit":
                            self.table.deal_card("dealer")
                        
                        # Score all hands
                        dealer_bust = self.table.dealer.hands[0].check_bust()
                        dealer_score = self.table.dealer.hands[0].get_score()
                        
                        for hand in player.hands:
                            if not hand.check_bust():
                                if dealer_bust:
                                    reward += 1
                                else:
                                    player_score = hand.get_score()
                                    if player_score > dealer_score:
                                        reward += 1
                                    elif player_score < dealer_score:
                                        reward -= 1
                        
                        self.episode_reward += reward
                        done = True
            
            observation = self._get_observation()
            info["last_action"] = action_str
            info["episode_reward"] = self.episode_reward
            info["current_hand_idx"] = player.current_hand_idx
            info["total_hands"] = len(player.hands)
            
            return observation, reward, done, info

    def render(self, mode="text"):
        print(self.table)

    def close(self):
        pass


if __name__ == "__main__":
    config = {
        "num_decks": 6,
        "red_card_position": 0.2,
        "bet_size": [1],
        "actions": ["stand", "hit", "double", "split"],
        "num_players": 1,
        "max_splits": 3,
        "split_aces_only_one_card": True
    }

    env = BlackjackEnv(config=config)

    for _ in range(20):
        print("-------------------- Starting round ...")
        observation = env.reset()
        print("Initial observation:", observation)
        true_count = observation["true_count"]
        print("True count:", true_count)

        bet = env.bet_space.sample()
        observation, reward, done, info = env.step(bet, action_type="bet")
        print(f"----- Bet: {env.bets[bet]}")
        print(f"New observation: {observation}")
        print(env.table.players[0])
        print(env.table.dealer)
        cumulative_reward = 0
        if not done:
            print("----- Making moves ...")
            moves = 0
            while not done and moves < 20:  # Safety limit
                moves += 1
                if observation["can_split"] == 1:
                    action = 3
                else:
                    action = env.move_space.sample()
                print(f"Action: {env.actions[action]}")
                if env.actions[action] == "split" and not env.table.can_player_split(0):
                    print("WARNING: Invalid split action!")
                    continue
                observation, reward, done, info = env.step(action, action_type="move")
                print(f"Info: {info}")
                print(f"New observation: {observation}")
                print(f"Reward: {reward}")
                print(env.table.players[0])
                
                cumulative_reward += reward
                if done:
                    print("Round ended.\n")
                    print(env.table.dealer)
                    break
            
            if moves >= 20:
                print("WARNING: Maximum moves reached!")

        print(f"Reward: {cumulative_reward}")
        print("-------------------- Terminated")