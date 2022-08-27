from agents.my_td import MyTDAgent
from agents.my_sarsa import MySARSAAgent


def main():
    # agent = MyTDAgent()
    agent = MySARSAAgent()
    agent.train()

if __name__ == '__main__':
    main()
