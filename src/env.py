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

    :param config: Configuration dictionary
    """

    def __init__(self, config):

        # Read the configuration file and set the parameters
        self.num_decks = config.get("num_decks", 6)
        self.cut_card_position = config.get("red_card_position", 0.2)
        self.bets = config.get("bet_size", [1])
        self.actions = config.get("actions", ["stand", "hit", "double"])
        self.num_players = config.get("num_players", 1)

        # Configure the action and observation spaces
        self.bet_space = gym.spaces.Discrete(len(self.bets))
        self.move_space = gym.spaces.Discrete(len(self.actions))
        self.observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(32),    # Player
                gym.spaces.Discrete(11),    # Dealer
                gym.spaces.Discrete(2),     # Ace
                gym.spaces.Box(             # Count
                    low=0,
                    high=1,
                    shape=(11, ),
                    dtype=float
                )
            )
        )

        self.table = Table(
            deck=Deck(num_decks=self.num_decks, cut_card_position=self.cut_card_position),
            num_players=self.num_players
        )

    def reset(self, seed=None):
        super().reset(seed=seed)

        # Initialize the environment state
        if self.table.deck.check_cut_card():
            self.table.deck.reset()
            self.table.reset_counter()
        self.table.reset()

        observation = (
            None,
            None,
            None,
            self.table.counter.true_count
        )
        return observation

    def step(self, action, action_type="move"):
        reward = 0
        done = False

        if action_type == "bet":
            bet = self.bets[action]
            self.table.players[0].bet = bet

            # Deal the cards and handle potential naturals
            self.table.deal()
            p_natural = self.table.players[0].hand.check_natural()
            d_natural = self.table.dealer.hand.check_natural()
            if p_natural or d_natural:
                if p_natural and d_natural:
                    reward = 0
                elif p_natural:
                    reward = 1.5
                else:
                    reward = -1
                done = True

            dealer_val = self.table.dealer.hand.cards[0].val
            dealer_val += 10 if dealer_val == 1 else 0
            observation = (
                self.table.players[0].hand.get_score(),
                dealer_val,
                int(self.table.players[0].hand.check_soft()),
                self.table.counter.true_count
            )
            return observation, reward, done

        if action_type == "move":
            if self.actions[action] == "hit":
                self.table.deal_card(player=0)
                if self.table.players[0].hand.check_bust():
                    reward = -1
                    done = True

            # TODO: add other actions
            elif self.actions[action] == "double":
                # bet = self.table.players[0].bet
                # self.table.players[0].bet = bet * 2
                self.table.deal_card(player=0)
                if self.table.players[0].hand.check_bust():
                    reward = -1
                    done = True
                else:
                    while self.table.dealer.move() == "hit":
                        self.table.deal_card("dealer")
                    if self.table.dealer.hand.check_bust():
                        reward = 1
                    else:
                        p_score = self.table.players[0].hand.get_score()
                        d_score = self.table.dealer.hand.get_score()
                        if p_score > d_score:
                            reward = 1
                        elif p_score < d_score:
                            reward = -1
                        else:
                            reward = 0
                    done = True
            else:
                while self.table.dealer.move() == "hit":
                    self.table.deal_card("dealer")
                if self.table.dealer.hand.check_bust():
                    reward = 1
                else:
                    p_score = self.table.players[0].hand.get_score()
                    d_score = self.table.dealer.hand.get_score()
                    if p_score > d_score:
                        reward = 1
                    elif p_score < d_score:
                        reward = -1
                    else:
                        reward = 0
                done = True

            dealer_val = self.table.dealer.hand.cards[0].val
            dealer_val += 10 if dealer_val == 1 else 0
            observation = (
                self.table.players[0].hand.get_score(),
                dealer_val,
                int(self.table.players[0].hand.check_soft()),
                self.table.counter.true_count
            )
            return observation, reward, done

    def render(self, mode="text"):
        print(self.table)

    def close(self):
        pass


if __name__ == "__main__":

    config = {
        "num_decks": 6,
        "red_card_position": 0.2,
        "bet_size": [1],
        "actions": ["stand", "hit"],
        "num_players": 1
    }

    env = BlackjackEnv(config=config)

    for _ in range(10):
        print("-------------------- Starting round ...")
        observation = env.reset()
        print("Initial observation:", observation)
        true_count = observation[3]
        print("True count:", true_count)

        bet = env.bet_space.sample()
        observation, reward, done = env.step(bet, action_type="bet")
        print(f"----- Bet: {env.bets[bet]}")
        print(f"New observation: {observation}")
        print(env.table.players[0])
        print(env.table.dealer)

        if not done:
            print("----- Making moves ...")
            while True:
                action = env.move_space.sample()
                print(f"Action: {env.actions[action]}")
                observation, reward, done = env.step(action, action_type="move")
                print(f"New observation: {observation}")
                print(env.table.players[0])
                if done:
                    print(env.table.dealer)
                    break

        print(f"Reward: {reward}")
        print("-------------------- Terminated")