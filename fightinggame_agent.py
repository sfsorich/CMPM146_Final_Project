#!/usr/bin/env python

import argparse
import retro
import numpy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#hyperparamters for Q learning
gamma = 0.95
learning_rate = 0.1
max_memory = 10000
# Exploration parameters for epsilon greedy strategy
epsilon = 1.0            # exploration probability at start
epsilon_min = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob
#number of episodes to run
episodes = 250


# stack_size = 4
# state_stack = deque()
# def stack_states(state, state_stack):

network = Sequential()
network.add(Dense(36, input_shape= (6,), activation='relu')) #basic input dimensions to neural net and layers
network.add(Dense(24, activation='relu'))
network.add(Dense(12, activation='sigmoid')) #output layer
network.compile(loss='mse', optimizer=Adam(lr=learning_rate)) #need to setup the learning functions for this to create network

memory = deque([], maxlen=max_memory)
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    global epsilon
    global network
    if numpy.random.rand() <= epsilon:
        return env.action_space.sample()
    epsilon -= decay_rate
    act_values = network.predict(state.reshape((1, -1)))
    act_values[act_values < 0.5] = 0
    act_values[act_values >= 0.5] = 1
    return act_values

def replay(batch_size):
    if len(memory) < batch_size:
        return
    minibatch = random.sample(memory, batch_size)
    for state, action, rew, next_state, done in minbatch:
        target = network.predict(state.reshape(1, -1))
        if done:
            target[0] = reward
        else:
            Q_next = max(network.predict(next_state.reshape(1, -1)))




def main():
    #given by random agent.py setting up gym retro usage
    parser = argparse.ArgumentParser()
    parser.add_argument('game', default='SuperStreetFighterIITurboRevival-GbAdvance', help='the name or path for the game to run')
    parser.add_argument('state', nargs='?', default=retro.State.DEFAULT, help='the initial state file to load, minus the extension')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    args = parser.parse_args()

    global env
    env = retro.make(args.game, args.state, players=args.players)
    #print(env.observation_space)


    training = True

    try:
        for ep in range(episodes):
            env.reset()
            action = env.action_space.sample() #this is where DQ function evaluates which action to take
            ob, rew, done, info = env.step(action) #take the action we think will result in a reward
            t = 0
            totrew = [0] * args.players

            state_old = numpy.array([info['x-p1'], info['y-p1'], info['x-p2'], info['y-p2'], info['state-p1'], info['state-p2']]) #this is our observation_space
            if ep < 0: #for the first 20 episodes take random actions to pretrain the network
                while True:
                    action = env.action_space.sample() #this is where DQ function evaluates which action to take
                    action_hold = numpy.zeros(12)
                    ob, rew, done, info = env.step(action) #take the action we think will result in a reward
                    state_new = numpy.array([info['x-p1'], info['y-p1'], info['x-p2'], info['y-p2'], info['state-p1'], info['state-p2']]) #this is our observation_space
                    print(numpy.multiply(rew, action).reshape((1,-1)))
                    remember(state_old, action, rew, state_new, done)
                    network.fit(state_old.reshape((1,-1)), numpy.multiply(rew, action).reshape((1,-1)), epochs=1)
                    state_old = state_new
                    env.render()
                    if done:
                        print("============== EPISODE {} FINISHED ==============".format(ep))
                        break

            else :
                while True:
                    action = act(state_old)
                    action_hold = numpy.zeros(12)
                    ob, rew, done, info = env.step(action) #take the action we think will result in a reward
                    if (info['health-p2'] < info['health-p1']): rew += 5.0
                    if (info['health-p2'] > info['health-p1']): rew -= 7.5
                    print(action)
                    print("REWARD: {}".format(rew))
                    network.fit(state_old.reshape((1,-1)), numpy.multiply(rew, action).reshape((1,-1)), epochs=1)
                    #print(action_hold)
                    env.render()
                    if done:
                        print("============== EPISODE {} FINISHED ==============".format(ep))
                        break

    except KeyboardInterrupt:
        exit(0)

if __name__ == '__main__':
    main()



    # def shared():
    #     t += 1
    #     if t % 10 ==0:
    #         print('INFO: {}'.format(info))
    #         timer = info['timer']
    #
    #     if args.players == 1:
    #         rew = [rew]
    #
    #     for i, r in enumerate(rew):
    #         totrew[i] += r
    #         if verbosity > 0:
    #             if r > 0:
    #                 print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
    #             if r < 0:
    #                 print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
    #     if done:
    #         env.render()
    #         try:
    #             if verbosity >= 0:
    #                 if args.players > 1:
    #                     print("done! total reward: time=%i, reward=%r" % (t, totrew))
    #                 else:
    #                     print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
    #                 input("press enter to continue")
    #                 print()
    #             else:
    #                 input("")
    #         except EOFError:
    #             exit(0)
