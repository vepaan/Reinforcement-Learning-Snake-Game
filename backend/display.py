from game import SnakeGameAI
from agent import Agent

agent = Agent()
agent.model.load("./model/model.pth")

game = SnakeGameAI(speed=20)

while True:
    state = agent.get_state(game)
    move = agent.get_action(state)
    reward, done, score = game.play_step(move)
    if done:
        game.reset()