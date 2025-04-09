import numpy as np


class Card:
    RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
    SUITS = ["C", "D", "H", "S"]

    def __init__(self, rank, suit):
        self.rank = rank
        self.suit = suit
        self.val = min(Card.RANKS.index(rank)+1, 10) if rank != "A" else 11

    def __str__(self):
        return f"{self.rank}{self.suit}"


class Deck:
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
        # noise = np.random.randint(-2*(self.num_decks+1), 2*(self.num_decks+1))
        # self.cut_card_position = int(
        #     (1-self.cut_card_position)*len(self.cards)
        # ) + noise
        self.cut_card_pos = int((1-self.cut_card_position)*len(self.cards))

    def draw(self):
        return self.cards.pop()

    def check_cut_card(self):
        return len(self) < self.cut_card_pos


class Hand:
    def __init__(self, cards=None):
        self.cards = cards if cards else []

    def __len__(self):
        return len(self.cards)

    def __str__(self):
        return " ".join(map(str, self.cards))

    def check_usable_ace(self):
        val, ace = 0, 0
        for card in self.cards:
            if card.val == 1:
                ace += 1
            val += min(card.val, 10)

        if ace > 0 and val + 10 <= 21:
            return True
        return False

    def get_value(self):
        val = sum(card.val for card in self.cards)
        if self.check_usable_ace():
            val += 10
        return val

    def check_natural(self):
        return self.get_value() == 21 and len(self.cards) == 2

    def check_bust(self):
        return self.get_value() > 21

    def get_score(self):
        score = 0
        if not self.check_bust():
            score = self.get_value()
        return score

    def reset(self):
        self.cards = []


class Player:
    def __init__(self, id, hand=None):
        self.id = id
        self.hand = hand if hand else Hand()

    def __str__(self):
        return f"Player {self.id}: {self.hand}"

    def move(self):
        # TODO: Implement the move
        pass

    def reset(self):
        self.hand.reset()


class Dealer(Player):
    def __init__(self, hand=None):
        super().__init__("dealer", hand)

    def __str__(self):
        return f"Dealer: {self.hand}"

    def move(self):
        if self.hand.get_value() < 17:
            return "hit"
        return "stand"


class Counter:
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
        elif card.val in [10, 11]:
            self.count -= 1

        # # Adjust the count for the number of decks remaining in the shoe
        # # TODO: Check this in the book
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
    def __init__(self, deck, num_players):
        self.deck = deck
        self.num_players = num_players
        self.counter = Counter(num_decks=deck.num_decks)
        self.players = {i: Player(i) for i in range(num_players)}
        self.dealer = Dealer()

    def __str__(self):
        return "TODO"

    def deal_card(self, player):
        card = self.deck.draw()
        if player == "dealer":
            self.dealer.hand.cards.append(card)
        else:
            self.players[player].hand.cards.append(card)
        self.counter.update(card)

    def deal(self):
        for i in range(self.num_players):
            for _ in range(2):
                self.deal_card(i)
        self.deal_card("dealer")
        card = self.deck.draw()
        self.dealer.hand.cards.append(card)

    def reset_counter(self):
        self.counter.reset()

    def reset(self):
        for player in self.players.values():
            player.reset()
        if self.dealer.hand.cards:
            self.counter.update(self.dealer.hand.cards[1])
        self.dealer.reset()


class BlackJackGame:
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