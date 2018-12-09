#!/usr/bin/env python

import argparse
import retro
from keras.models import Sequential

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('game', help='the name or path for the game to run')
    parser.add_argument('state', nargs='?', help='the initial state file to load, minus the extension')
    parser.add_argument('--scenario', '-s', default='scenario', help='the scenario file to load, minus the extension')
    parser.add_argument('--record', '-r', action='store_true', help='record bk2 movies')
    parser.add_argument('--verbose', '-v', action='count', default=1, help='increase verbosity (can be specified multiple times)')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='decrease verbosity (can be specified multiple times)')
    parser.add_argument('--players', '-p', type=int, default=1, help='number of players/agents (default: 1)')
    args = parser.parse_args()

    env = retro.make(args.game, args.state or retro.State.DEFAULT, scenario=args.scenario, record=args.record, players=args.players)
    verbosity = args.verbose - args.quiet

    model = Sequential()
    model.add(Dense(24, input_dim = 4, activation='relu'))
    model.add(Dense(12, activation='relu')) #output layer
    #model.compile() #need to setup the learning functions for this to create network
    print(env.observation_space)

    try:
        while True:
            ob = env.reset()
            t = 0
            totrew = [0] * args.players
            timer = 99
            while True:
                ac = env.action_space.sample()
                ob, rew, done, info = env.step(ac)
                if timer != info['timer']:
                    print('INFO: {}'.format(info))
                timer = info['timer']

                t += 1
                if t % 10 == 0:
                    if verbosity > 1:
                        infostr = ''
                        if info:
                            infostr = ', info: ' + ', '.join(['%s=%i' % (k, v) for k, v in info.items()])
                        print(('t=%i' % t) + infostr)
                    env.render()
                if args.players == 1:
                    rew = [rew]
                for i, r in enumerate(rew):
                    totrew[i] += r
                    if verbosity > 0:
                        if r > 0:
                            print('t=%i p=%i got reward: %g, current reward: %g' % (t, i, r, totrew[i]))
                        if r < 0:
                            print('t=%i p=%i got penalty: %g, current reward: %g' % (t, i, r, totrew[i]))
                if done:
                    env.render()
                    try:
                        if verbosity >= 0:
                            if args.players > 1:
                                print("done! total reward: time=%i, reward=%r" % (t, totrew))
                            else:
                                print("done! total reward: time=%i, reward=%d" % (t, totrew[0]))
                            input("press enter to continue")
                            print()
                        else:
                            input("")
                    except EOFError:
                        exit(0)
                    break
    except KeyboardInterrupt:
        exit(0)

if __name__ == '__main__':
    main()
