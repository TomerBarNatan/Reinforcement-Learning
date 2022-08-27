import random
from abc import abstractmethod
import pprint
from environment import Environment
states = list(range(4, 23))


class MyTDAgent:
    """
    0 - stick
    1 - hit
    """

    def init_pi(self):
        self.pi = {possible_sum: 'stand' for possible_sum in states if possible_sum>=18}
        self.pi.update({possible_sum: 'hit' for possible_sum in states if possible_sum<18})


    def init_q(self):
        for state in states:
            self.q[(state, 0)] = (0.0, 0)  # average, count
            self.q[(state, 1)] = (0.0, 0)

    def init_v(self):
        self.v = {possible_sum: 0. for possible_sum in states}

    def __init__(self):
        self.epsilon = 0.05
        self.env = Environment()
        # self.eta = 0.00005
        self.eta = 0.0005
        self.gamma = 1.
        self.v = dict()  # approximate state-action value
        self.init_v()
        self.pi = dict()  # policy
        self.init_pi()

    def policy(self, state):
        action = self.pi[state]
        return action

    def step(self, action):
        if action == 'stand':
            self.env.stick()
        else:
            self.env.hit()

        state, done =  self.env.get_state()
        return state, done, self.env.get_reward()

    def run_episode(self):
        self.env.deal_dealer()
        _, _, _ = self.step('hit')
        current_state, done, _ = self.step('hit')
        while not done:
            action = self.policy(current_state)
            next_state, done, reward = self.step(action)
            self.update_v(current_state, reward, next_state, done)
            current_state = next_state


    def update_v(self, s, r, s_next, done):
        if not done:
            self.v[s] = self.v[s] + self.eta*(r+self.gamma*self.v[s_next]-self.v[s])
        else:
            self.v[s] = self.v[s] + self.eta*(r-self.v[s])

    def train(self):
        # episodes = 35000000
        episodes = 3000000
        # print (self.calculate_sum_prob(20))
        for i in range(episodes):
            self.run_episode()
            self.env = Environment()
            if i % 100000 == 0:
                print (i, self.v[4])
        self.combined_probs = {sum:self.calculate_sum_prob(sum) * estimated_prob for sum, estimated_prob in self.v.items()}
        self.winning_prob = sum([self.calculate_sum_prob(sum) * estimated_prob for sum, estimated_prob in self.v.items()])
        pprint.pprint(self.v)
        pprint.pprint(self.combined_probs)
        print(f'total winning prob: {self.winning_prob}')

    def calculate_sum_prob(self, sum):
        prob = 0
        min_value = max(sum-11, 2)
        max_value = min (sum-2, 11)
        for j in range(min_value, max_value+1):
            deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
            first_prob = deck.count(j)/len(deck)
            deck.remove(j)
            second_prob = deck.count(sum-j)/len(deck)
            prob += first_prob*second_prob
        return prob