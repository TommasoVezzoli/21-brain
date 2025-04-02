import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Union, Tuple, Dict, Any

class BlackjackMultiDeckEnv(gym.Env):
    """
    Blackjack environment with multiple decks and no replacement.
    Extends the Gymnasium Blackjack environment to support a custom number of decks
    and card drawing without replacement.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(
        self,
        num_decks: int = 6,
        natural: bool = False,
        sab: bool = False,
        render_mode: Optional[str] = None,
    ):
        self.num_decks = num_decks
        self.natural = natural  # Natural blackjack gives 1.5x reward
        self.sab = sab  # Sab rule: dealer hits on soft 17
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(2)  # Hit or Stand
        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(32),  # Player's sum (0-31)
                spaces.Discrete(11),  # Dealer's showing card (1-10)
                spaces.Discrete(2),   # Player has usable ace (0 or 1)
                spaces.Box(          # Card count information (optional)
                    low=0, 
                    high=1, 
                    shape=(13,), 
                    dtype=np.float32
                )
            )
        )
        
        self.render_mode = render_mode
        
        # Initialize the deck
        self.reset_deck()
        
    def reset_deck(self):
        """Initialize a new deck with the specified number of decks"""
        # A single deck has 4 suits × 13 ranks = 52 cards
        # Card values: A=1, 2-10=face value, J,Q,K=10
        card_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # A,2,3,4,5,6,7,8,9,10,J,Q,K
        # Create a deck with the correct number of each card
        self.deck = []
        for _ in range(self.num_decks):
            for suit in range(4):  # 4 suits
                for value in card_values:
                    self.deck.append(value)
        
        # Shuffle the deck
        np.random.shuffle(self.deck)
        
        # Track card counts for observation
        self.card_count = {i: self.num_decks * 4 for i in range(1, 11)}
        self.card_count[10] = self.num_decks * 16  # 10, J, Q, K all count as 10
    
    def draw_card(self):
        """Draw a card from the deck without replacement"""
        if not self.deck:
            # If deck is empty, reset it (like adding new decks in a casino)
            self.reset_deck()
        
        card = self.deck.pop()
        # Update card count
        self.card_count[min(card, 10)] -= 1
        return card
    
    def draw_hand(self):
        """Draw an initial hand (two cards)"""
        return [self.draw_card(), self.draw_card()]
    
    def usable_ace(self, hand):
        """Determine if the hand has a usable ace (can count as 11 without busting)"""
        # Count Aces as 11 if it doesn't bust the hand
        val, aces = 0, 0
        for card in hand:
            if card == 1:  # Ace
                aces += 1
            val += min(card, 10)  # Face cards count as 10
            
        # Add 10 if there's an ace and it doesn't bust
        if aces > 0 and val + 10 <= 21:
            return True
        return False
    
    def sum_hand(self, hand):
        """Calculate the sum of a hand, accounting for usable aces"""
        if self.usable_ace(hand):
            return sum(min(card, 10) for card in hand) + 10
        else:
            return sum(min(card, 10) for card in hand)
    
    def is_bust(self, hand):
        """Check if a hand is bust (sum > 21)"""
        return self.sum_hand(hand) > 21
    
    def score(self, hand):
        """Score a hand; return 0 if bust"""
        return 0 if self.is_bust(hand) else self.sum_hand(hand)
    
    def is_natural(self, hand):
        """Check for a natural blackjack (21 with 2 cards)"""
        return len(hand) == 2 and self.sum_hand(hand) == 21
    
    def get_normalized_card_counts(self):
        """Get normalized card counts for observation"""
        # Convert counts to probabilities
        total_cards = sum(self.card_count.values())
        normalized = np.zeros(13)
        for i in range(1, 11):
            normalized[i-1] = self.card_count[i] / total_cards
        return normalized
        
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment for a new game"""
        super().reset(seed=seed)
        
        # If the deck is getting low, reset it
        if len(self.deck) < 15:  # Arbitrary threshold
            self.reset_deck()
            
        # Deal initial cards
        self.dealer = self.draw_hand()
        self.player = self.draw_hand()
        
        # # Check for naturals
        # dealer_natural = self.is_natural(self.dealer)
        # player_natural = self.is_natural(self.player)
        
        # # Handle naturals
        # if self.natural and (player_natural or dealer_natural):
        #     # Game is over if either has a natural
        #     if player_natural and dealer_natural:
        #         reward = 0
        #     elif player_natural:
        #         reward = 1.5
        #     else:
        #         reward = -1
        #     done = True
        #     self.done = True
        # else:
        #     # Game continues
        #     reward = 0
        #     done = False
        #     self.done = False
        
        # Get the observation
        observation = (
            self.sum_hand(self.player),
            self.dealer[0],  # Dealer's upcard
            int(self.usable_ace(self.player)),
            self.get_normalized_card_counts()
        )
        
        # Set up info dict
        info = {"player_hand": self.player, "dealer_hand": self.dealer}
        
        return observation, info
    
    def step(self, action):
        """Take a step in the environment based on the action"""
        
        assert self.action_space.contains(action)
        
        # Check for naturals
        dealer_natural = self.is_natural(self.dealer)
        player_natural = self.is_natural(self.player)
        
        # Handle naturals
        if self.natural and (player_natural or dealer_natural):
            # Game is over if either has a natural
            if player_natural and dealer_natural:
                reward = 0

            #if the player has a natural blackjack and the dealer does not, the player wins
            #according to the Sab rule, the reward is not different to a normal one
            elif player_natural and self.sab:
                reward = 1
            #if the player has a natural and sab is false and natural is true we
            #give more importance to the player's natural
            elif player_natural and self.natural:
                reward = 1.5
            else:
                reward = -1
            self.done = True
        else:
            # Game continues
            reward = 0
            self.done = False

        if self.done:
            # Game already over due to natural
            observation = (
                self.sum_hand(self.player),
                self.dealer[0],
                int(self.usable_ace(self.player)),
                self.get_normalized_card_counts()
            )
            reward = 0 
            terminated = True
            truncated = False
            info = {"player_hand": self.player, "dealer_hand": self.dealer}
            return observation, reward, terminated, truncated, info
        
        # Player's turn
        if action == 1:  # hit
            self.player.append(self.draw_card())
            if self.is_bust(self.player):
                terminated = True
                reward = -1
            else:
                terminated = False
                reward = 0
        else:  # stand
            terminated = True
            
            # Dealer's turn
            while not self.is_bust(self.dealer):
                # Dealer hits below 17, or on soft 17 if sab is True
                score = self.sum_hand(self.dealer)
                if score < 17 or (self.sab and score == 17 and self.usable_ace(self.dealer)):
                    self.dealer.append(self.draw_card())
                else:
                    break
                    
            # Determine the winner
            player_score = self.score(self.player)
            dealer_score = self.score(self.dealer)
            
            if self.is_bust(self.dealer):
                reward = 1
            elif player_score > dealer_score:
                reward = 1
            elif player_score < dealer_score:
                reward = -1
            else:
                reward = 0  # Push (tie)
        
        # Get the new observation
        observation = (
            self.sum_hand(self.player),
            self.dealer[0],
            int(self.usable_ace(self.player)),
            self.get_normalized_card_counts()
        )
        
        # For gymnasium v0.28+
        truncated = False
        
        info = {"player_hand": self.player, "dealer_hand": self.dealer}
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment for human viewing"""
        if self.render_mode == "human":
            print(f"Player: {self.player} (Sum: {self.sum_hand(self.player)})")
            print(f"Dealer: {self.dealer} (Sum: {self.sum_hand(self.dealer)})")
            
    def close(self):
        """Close the environment"""
        pass

# Example usage:
if __name__ == "__main__":
    env = BlackjackMultiDeckEnv(num_decks=6)
    observation, info = env.reset()
    
    print("Initial observation:", observation)
    print("Player hand:", info["player_hand"])
    print("Dealer upcard:", info["dealer_hand"][0])
    
    # Take some actions
    for _ in range(3):
        action = env.action_space.sample()  # Random action
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {'Hit' if action == 1 else 'Stand'}")
        print(f"New observation: {observation}")
        print(f"Reward: {reward}")
        print(f"Game over: {terminated}")
        
        if terminated:
            break