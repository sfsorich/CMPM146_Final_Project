import retro
def main():
	#StreetFighterIISpecialChampionEdition-Genesis
    env = retro.make(game='SuperStreetFighterIITurboRevival-GbAdvance', state='RyuVSKen-Level8-Normal.state')
    obs = env.reset()
    #print("obs" + str(obs))
    print(env.action_space.n)# -> 12
    #print(env.observation_space) -> (160, 240, 3) 
    #env.get_action_meanings()
    '''
    i = 0
    while True:
    	i = i+1
    	print(i)
    	env.render()
    	env.step(env.action_space.sample()) # take a random action
    	print(info)
    '''
    while True:
        hp1 = 176
        hp2 = 176
        obs, rew, done, info = env.step(env.action_space.sample())
        #print(info['health-p1']) # -> getting values from data.json
        #print(env.action_space.sample()) # -> [1 1 1 0 0 1 1 0 0 1 0 0]
        #env.unwrapped.get_action_meanings() #-> 
        print(rew)
        if hp2 > info['health-p2']:
            #print("HITTTTTTTTTTTTTTTT")            
            hp2 = info['health-p2']
            rew += 2
        print(hp2)
        
        print('rew', rew)

        if hp1 < info['health-p1']:
            hp1 = info['health-p1']
            rew -= 1
        env.render()
        print(hp1)

        print(rew)
        #if info['health-p1'] == 0:
            #done = True
        if done:
            obs = env.reset()
	
if __name__ == '__main__':
    main()
