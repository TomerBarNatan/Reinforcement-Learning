import random
from abc import abstractmethod
import pprint
from environment import Environment
states = list(range(4, 23))

class MySARSAAgent:
    """
    0 - stick
    1 - hit
    """

    def init_pi(self):
        self.pi = {possible_sum: 'stand' for possible_sum in states if possible_sum >= 18}
        self.pi.update({possible_sum: 'hit' for possible_sum in states if possible_sum < 18})


    def init_q(self):
        self.q = {}
        for possible_sum in states:
            self.q.update({possible_sum:{action:0 for action in ['stand','hit']}})

    def init_v(self):
        self.v = {possible_sum: 0. for possible_sum in states}

    def __init__(self):
        self.epsilon = 0.15
        self.env = Environment()
        self.eta = 0.00005
        self.gamma = 1.
        self.v = dict()  # approximate state-action value
        self.init_v()
        self.pi = dict()  # policy
        self.init_pi()
        self.init_q()

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

    def get_action(self, next_state):
        rand = random.uniform(0,1)
        if rand > self.epsilon:
            next_action = max(self.q[next_state], key=self.q[next_state].get)
        else:
            next_action = random.choice(['hit', 'stand'])
        return next_action

    def run_episode(self):
        self.env.deal_dealer()
        _, _, _ = self.step('hit')
        current_state, done, _ = self.step('hit')
        current_action = random.choice(['hit','stand'])
        next_action = None
        while not done:
            next_state, done, reward = self.step(current_action)
            if not done:
                next_action = self.get_action(next_state)
            self.update_q(current_state, current_action, next_state, next_action, reward, done)
            current_state = next_state

    def update_q(self, s, a, s_next, a_next, r, done):
        if not done:
            self.q[s][a] = self.q[s][a] + self.eta*(r+self.gamma*self.q[s_next][a_next]-self.q[s][a])
        else:
            self.q[s][a] = self.q[s][a] + self.eta*(r-self.q[s][a])

    def train(self):
        # episodes = 35000000
        episodes = 6000000
        # episodes = 200
        # print (self.calculate_sum_prob(20))
        for i in range(episodes):
            self.run_episode()
            self.env = Environment()
            if i % 100000 == 0:
                print (i, self.q[8])
        self.optimal_policy = {state : (max(self.q[state], key=self.q[state].get),self.q[state][max(self.q[state], key=self.q[state].get)]) for state in self.q.keys()}
        self.winning_prob = {sum : self.calculate_sum_prob(sum) * estimated_prob for sum, (_,estimated_prob) in self.optimal_policy.items()}
        self.total_winning_prob = sum([self.calculate_sum_prob(sum) * estimated_prob for sum, (_,estimated_prob) in self.optimal_policy.items()])
        pprint.pprint(self.optimal_policy)
        pprint.pprint(self.winning_prob)
        print(self.total_winning_prob)

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