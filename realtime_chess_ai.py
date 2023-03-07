#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:03:57 2023

@author: zc
"""

import classic.chess_rt as chess
import numpy as np

if __name__ == "__main__":
    # create an environment
    env = chess.env('ansi')
    # reset the environment
    obs = env.reset()
    count_down = 400
    
    for agent in env.agent_iter():
        print(agent)
        env.unwrapped.set_color(agent)
        """
        TODO: Understand observation["observation"]
        
        Understanding the 8*8*(P+L) list maybe useful for the model training, 
        but the list is needed to confirm the values are correct, since it now
        records by pieces instead of only 2 players the original intention.
        """
        observation, reward, termination, truncation, info = env.last()
        if termination:
            print("Game Over")
            break
        if truncation:
            env.step(None)
            continue
        
        """
        TODO: Cool down mechanism
        
        Could use the api of KungFu chess or implement it ourself.
        """
        # Filter out invalid actions
        valid_actions = np.where(np.array(observation["action_mask"]) == 1)[0]
        """
        TODO: Model training
        
        Machine learning part should replace the following line to choose an 
        action itself.
        """
        # Random choose a valid action
        action = np.random.choice(valid_actions)
        
        col = action//(8*74)
        row = (action//74)%8
        print("action index:", action)
        # The coordination is a subjective perspective to the player side, 
        # black side's position is mirrored vertically.
        print("move (x = {}, y = {}, c = {})".format(col, row, action-(col*8+row)*74))
        env.step(action)
        pvs_agent = agent
        # print("reward:", reward)
        print(env.render())
        print()
    
        count_down -= 1
        if count_down == 0:
            break
    