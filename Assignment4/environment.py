import random


class Environment:

    def __init__(self):
        self.sum_player = 0
        self.sum_dealer = 0
        self.deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10,10, 11] * 4
        self.player_sticked = False

    def get_card(self):
        """
        Returns a random card with value from 1 to 10.
        """
        idx = random.choice(range(len(self.deck)))
        card = self.deck.pop(idx)
        return card


    def deal_dealer(self):
        """
        Deals cards to dealer until 16 or less.
        Remembers first card of the dealer from 2 to 11.
        """
        while self.sum_dealer < 16:
            self.sum_dealer = self.get_next_state(
                self.sum_dealer
            )

    def is_end(self):
        """
        Checks if it is the end of the game.
        """
        return self.sum_player > 21 or self.player_sticked

    def get_next_state(self, sum_cards):
        """
        Go to the next state with with given sum and usable ace info.
        """
        card = self.get_card()
        sum_cards += card
        return sum_cards

    def stick(self):
        """
        Mark that player sticked.
        """
        self.player_sticked = True

    def get_reward(self):
        """
        Return the reward for the game.
        """
        if self.player_sticked:
            # if self.sum_player < 18:
            #     return 0.
            if self.sum_player > 21:
                return 0.
            else:
                if self.sum_dealer > 21:
                    return 1.
                elif self.sum_player > self.sum_dealer:
                    return 1.
                else:
                    return 0.
        else:
            return 0

    def hit(self):
        """
        Update state of the player.
        """
        self.sum_player = self.get_next_state(
            self.sum_player
        )

    def get_state(self):
        """
        Get state of the player.
        """
        return self.sum_player, self.is_end()