import numpy as np


class Card:
    """
    Class representing a playing card.
    """
    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    SUITS = ["C", "D", "H", "S"]

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.val = min(Card.RANKS.index(rank)+1, 10)

    def __str__(self):
        return f"{self.rank}{self.suit}"


class Deck:
    """
    Class representing a deck of cards.
    """
    def __init__(self, num_decks, cut_card_position=0.25):
        self.num_decks = num_decks
        self.cut_card_position = cut_card_position
        self.cards = []
        self.cut_card_pos = None
        self.reset()

    def __len__(self):
        return len(self.cards)

    def reset(self):
        # Reset the deck
        self.cards = [Card(rank, suit) for rank in Card.RANKS for suit in Card.SUITS]*self.num_decks
        np.random.shuffle(self.cards)

        # Reset the cut card position
        self.cut_card_pos = int((1-self.cut_card_position)*len(self.cards))

    def draw(self):
        return self.cards.pop()

    def check_cut_card(self):
        return len(self.cards) <= self.num_decks * 52 * self.cut_card_position


class Hand:
    """
    Class representing a player's hand.
    """
    def __init__(self, cards=None):
        self.cards = cards if cards else []
        self.is_split = False
        self.is_done = False

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return " ".join(map(str, self.cards))

    def check_soft(self):
        val, ace = 0, 0
        for card in self.cards:
            if card.rank == "A":
                ace += 1
            val += card.val

        if ace > 0 and val+10 <= 21:
            return True
        return False

    def get_value(self):
        val = sum(card.val for card in self.cards)
        if self.check_soft():
            val += 10
        return val

    def check_natural(self):
        return self.get_value() == 21 and len(self.cards) == 2 and not self.is_split

    def check_bust(self):
        return self.get_value() > 21

    def get_score(self):
        score = 0
        if not self.check_bust():
            score = self.get_value()
        return score
    
    def can_split(self):
        if len(self.cards) != 2:
            return False
        return self.cards[0].rank == self.cards[1].rank
    
    def split(self):
        if not self.can_split():
            raise ValueError("Cannot split hand")
        self.is_split = True
        new_hand = Hand(cards=[self.cards.pop()])
        new_hand.is_split = True
        self.is_split = True

        return new_hand

    def reset(self):
        self.cards = []
        self.is_split = False
        self.is_done = False


class Player:
    """
    Class representing a player in the game.
    """
    def __init__(self, id, hand=None):
        self.id = id
        # self.hand = hand if hand else Hand()
        self.hands = [hand if hand else Hand()]
        self.current_hand_idx = 0
        self.can_split_aces = True

    def __str__(self):
        # return f"Player {self.id}: {self.hand}"
        hands_str = "\n  ".join([f"Hand {i}: {hand}" for i, hand in enumerate(self.hands)])
        return f"Player {self.id}:\n  {hands_str}"

    def move(self):
        pass

    def get_current_hand(self):
        if self.current_hand_idx < len(self.hands):
            return self.hands[self.current_hand_idx]
        return None
    
    def move_to_next_hand(self):
        if self.current_hand_idx < len(self.hands) - 1:
            self.current_hand_idx += 1
            return True
        return False
    
    def move_to_previous_hand(self):
        if self.current_hand_idx > 0:
            self.current_hand_idx -= 1
            return True
        return False

    def split_current_hand(self):
        current_hand = self.get_current_hand()
        if current_hand and current_hand.can_split():
            if current_hand.cards[0].rank == "A" and not self.can_split_aces:
                raise ValueError("Cannot split aces again")
            
            new_hand = current_hand.split()
            if new_hand:
                self.hands.insert(self.current_hand_idx + 1, new_hand)
                return True
        return False

    def reset(self):
        self.hands =  [Hand()]
        self.current_hand_idx = 0


class Dealer(Player):
    """
    Class representing the dealer in the game.
    """
    def __init__(self, hand=None):
        super().__init__("dealer", hand)

    def __str__(self):
        return f"Dealer: {self.hands[0]}"

    def move(self):
        if self.hands[0].get_value() < 17:
            return "hit"
        return "stand"


class Counter:
    """
    Class representing the card counter for the game.
    """
    def __init__(self, num_decks):
        self.num_decks = num_decks
        self.cards = []
        self.cards_seen = 0
        self.count = 0
        self.true_count = 0

    def __str__(self):
        return f"Count: {self.count} | True Count: {self.true_count}"

    def update(self, card):
        if card.val in [2, 3, 4, 5, 6]:
            self.count += 1
        elif card.val == 10 or card.rank == "A":
            self.count -= 1

        # # Adjust the count for the number of decks remaining in the shoe
        self.cards.append(card)
        self.cards_seen += 1
        decks_remaining = max(0.5, (self.num_decks * 52 - self.cards_seen) / 52)
        self.true_count = self.count / decks_remaining

    def reset(self):
        self.count = 0
        self.true_count = 0
        self.cards = []
        self.cards_seen = 0


class Table:
    """
    Class representing the table in the game.
    """
    def __init__(self, deck, num_players):
        self.deck = deck
        self.num_players = num_players
        self.counter = Counter(num_decks=deck.num_decks)
        self.players = {i: Player(i) for i in range(num_players)}
        self.dealer = Dealer()
        self.max_splits = 3

    def __str__(self):
        players_str = "\n".join([str(player) for player in self.players.values()])
        return f"{players_str}\n{self.dealer}\nCounter: {self.counter}"

    def deal_card(self, player, hand_idx=None):
        card = self.deck.draw()
        if player == "dealer":
            self.dealer.hands[0].cards.append(card)
        else:
            if hand_idx is None:
                self.players[player].get_current_hand().cards.append(card)
            else:
                self.players[player].hands[hand_idx].cards.append(card)
        self.counter.update(card)
        return card

    def deal(self):
        for i in range(self.num_players):
            for _ in range(2):
                self.deal_card(i, hand_idx=0)
        self.deal_card("dealer")
        card = self.deck.draw()
        self.dealer.hands[0].cards.append(card)

    def reset_counter(self):
        self.counter.reset()

    def reset(self):
        for player in self.players.values():
            player.reset()
        if self.dealer.hands[0].cards:
            self.counter.update(self.dealer.hands[0].cards[1])
        self.dealer.reset()
    
    def can_player_split(self, player_id):
        player = self.players[player_id]
        current_hand = player.get_current_hand()

        if current_hand and current_hand.can_split():
            split_count = sum(1 for hand in player.hands if hand.is_split)
            return split_count < self.max_splits
        return False


class BlackJackGame:
    """
    Class representing the Blackjack game. Used for environment simulation.
    """
    def __init__(
            self,
            num_decks=6,
            cut_card_position=0.2,
            actions=None,
            num_players=1
    ):
        self.deck = Deck(num_decks=num_decks, cut_card_position=cut_card_position)
        self.actions = actions if actions else ["hit", "stand"]
        self.n_actions = len(self.actions)
        self.table = Table(deck=self.deck, num_players=num_players)

    def play_round(self, verbose=False):
        self.table.deal()

        if verbose:
            for player in self.table.players.values():
                print(player)
            print(self.table.dealer)
            print(self.table.counter)
            print("----- Making moves ...")

        for player in self.table.players.values():
            player.move()
        while self.table.dealer.move() == "hit":
            self.table.deal_card("dealer")

        if verbose:
            for player in self.table.players.values():
                print(player)
            print(self.table.dealer)
            print(self.table.counter)
            print("----- Scoring ...")

        # Reset round
        self.table.reset()

    def play(self, n_rounds=10, verbose=False):
        for i in range(n_rounds):
            if verbose:
                print(f"-------------------- Starting round {str(i+1)} ...")
            self.play_round(verbose=verbose)
            if verbose:
                print(f"-------------------- Terminated")
            if self.deck.check_cut_card():
                self.deck.reset()

if __name__ == "__main__":
    game = BlackJackGame()
    game.play(n_rounds=1, verbose=True)