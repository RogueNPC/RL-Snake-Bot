import torch 
import random 
import numpy as np
from collections import deque
from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from model import Linear_QNet,QTrainer
from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         


    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self,game):
        head = game.snake[0]
        point_l=Point(head.x - BLOCK_SIZE, head.y)
        point_r=Point(head.x + BLOCK_SIZE, head.y)
        point_u=Point(head.x, head.y - BLOCK_SIZE)
        point_d=Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d))or
            (dir_l and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_r)),

            # Danger right
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_u and game.is_collision(point_u))or
            (dir_d and game.is_collision(point_d)),

            #Danger Left
            (dir_u and game.is_collision(point_r))or
            (dir_d and game.is_collision(point_l))or
            (dir_r and game.is_collision(point_u))or
            (dir_l and game.is_collision(point_d)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.food.x < game.head.x, # food is in left
            game.food.x > game.head.x, # food is in right
            game.food.y < game.head.y, # food is up
            game.food.y > game.head.y  # food is down
        ]
        return np.array(state,dtype=int)

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if (len(self.memory) > BATCH_SIZE):
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    # TODO: What is the role of epsilon in this method? Feel free to reference the OpenAI Gym RL tutorial from 02/09/22
    """ Epsilon introduces the ability for the agent to make more random moves instead of always choosing the action 
        it thinks is the best based on previously learned Q-values.  As the number of games the agent experiences--as 
        I'm guessing that's what self.n_game represents-- the value of epsilon decreases which, in turn, decreases 
        the probablity of the agent exploring (choosing a random action). """
    def get_action(self,state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if(random.randint(0,200)<self.epsilon):
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float).cpu()
            prediction = self.model(state0).cpu() # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

# TODO: Write a couple sentences describing the training process coded below.
""" At the start of a new game, the agent gets its starting state in the environment (the game).
    The agent then calculates its move through the get_action method and performs its move through
    the environment's get_state method.  The agent gets its new state after the move and then procedes
    to be rewarded positively, negatively, or neutrally on it's current action based on its behavior 
    through the train_short_memory method.  The agent then adds this to their memory which will be accessed
    when the game ends and the agent uses the train_long_memory method and learns through the tens of thousands
    of actions it performed in their previous game(s).  The training process will continue infinitly as the
    agent performs more actions and plays many more games of Snake. """
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    while True:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            # Train long memory,plot result
            game.reset()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > record): # new High score 
                record = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            plot_mean_scores.append(mean_score)
            plot(plot_scores,plot_mean_scores)


if(__name__=="__main__"):
    train()

# TODO: Write a brief paragraph on your thoughts about this implementation. 
# Was there anything surprising, interesting, confusing, or clever? Does the code smell at all?
""" Seeing the implementation of Epsilon was really eye-opening to me because my previous project, 
    which invovled a min-max algorithmic implementation of a chess-playing program, had the problem of
    prefering the same set of moves due to a lack of randomness.  Knowing about this implementation
    inspires me to go back and revise the chess-playing program to become more human-like using epsilon
    or another kind of randomness that fits within my implementation.
    
    The implementation of the agent as a whole was a lot more straight-forward than I thought--even taking
    into account the simplisity of Snake compared to games that are way more complex such as Starcraft.
    The idea of generating states, granting a reward, and having the agent learn through short-term and long-term
    methods only took up a few lines of code, its still way below my expectations of what is required to build a 
    reinforcement learning AI, no matter how simple it may be.  Taking these thoughts into a broader outlook, 
    I can almost guess this is the kind of structure that reinforcement learning neural networks such as 
    Leela Chess Zero and other such game-playing bots utilize. """