import gym
import numpy as np
import time
import matplotlib.pyplot as plt

def agent(ot, w):
    return 1 if ot @ w >= 0 else 0


def run_episode(env, w):
    total_reward = 0
    ot = env.reset()
    for i in range(201):
        # env.render()
        action = agent(ot, w)
        ot, reward, done, info = env.step(action)
        # time.sleep(0.01)
        total_reward += reward
        if done:
            break
    return total_reward


def random_search(env):
    for i in range(1000):
        w = np.random.uniform(low=-1, high=1, size=(4,))
        result = run_episode(env, w)
        if result >= 200:
            return i
    return 1000

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    results = []
    for i in range(1000):
        results.append(random_search(env))
        print(f"finished {i}")
    print(f"Avg number of episodes: {np.average(results)}")
    plt.hist(results, bins = 30)
    plt.title("Histogram of Random Search")
    plt.xlabel("Episodes required to reach 200 reward")
    plt.ylabel("Num of runs")
    plt.savefig(f"Histogram of Random Search.png")
    plt.show()
