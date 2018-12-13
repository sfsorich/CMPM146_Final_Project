#!/usr/bin/env python

import argparse
import retro
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#hyperparamters for Q learning
gamma = 0.95
learning_rate = 0.001
max_memory = 10000
# Exploration parameters for epsilon greedy strategy
explore_start = 1.0            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability
decay_rate = 0.0001            # exponential decay rate for exploration prob
#number of episodes to run
episodes = 250


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


def main():
    #given by random agent.py setting up gym retro usage
    parser = argparse.ArgumentParser()
    parser.add_argument('game', default='SuperStreetFighterIITurboRevival-GbAdvance', help='the name or path for the game to run')
    parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    args = parser.parse_args()

    env = retro.make(args.game, args.state or retro.State.DEFAULT, players=args.players)



    network = Sequential()
    network.add(Dense(36, input_shape= (6,), activation='relu')) #basic input dimensions to neural net and layers
    network.add(Dense(12, activation='sigmoid')) #output layer
    network.compile(loss='mse', optimizer=Adam(lr=learning_rate)) #need to setup the learning functions for this to create network
    #print(env.observation_space)
    training = True

    try:
        for ep in range(episodes):
            env.reset()
            action = env.action_space.sample() #this is where DQ function evaluates which action to take
            ob, rew, done, info = env.step(action) #take the action we think will result in a reward
            t = 0
            totrew = [0] * args.players
            if ep < 5: #for the first 20 episodes take random actions to pretrain the network
                while True:
                    action = env.action_space.sample() #this is where DQ function evaluates which action to take
                    action_hold = numpy.zeros(12)
                    #print(action_hold)
                    #print(info)
                    state = numpy.array([info['x-p1'], info['y-p1'], info['x-p2'], info['y-p2'], info['state-p1'], info['state-p2']]) #this is our observation_space
                    #print(state.shape)
                    #print(state.reshape((1,-1)))
                    network.fit(state.reshape((1,-1)), action_hold.reshape((1,-1)), epochs=1)
                    ob, rew, done, info = env.step(action) #take the action we think will result in a reward
                    #env.render()
                    if done:
                        break

            else :
                while True:

                    state =numpy.array( [info['x-p1'], info['y-p1'], info['x-p2'], info['y-p2'], info['state-p1'], info['state-p2']])
                    action_hold = numpy.zeros(12)

                    #network.predict(state)
                    action = network.predict(state.reshape(1, -1))
                    print(action)
                    action[action < 0.5] = 0
                    action[action >= 0.5] = 1
                    print(action)
                    ob, rew, done, info = env.step(action[0]) #take the action we think will result in a reward
                    network.fit(state.reshape((1,-1)), action_hold.reshape((1,-1)), epochs=1)
                    print(action_hold)
                    env.render()
                    if done:
                        break

    except KeyboardInterrupt:
        exit(0)

if __name__ == '__main__':
    main()
