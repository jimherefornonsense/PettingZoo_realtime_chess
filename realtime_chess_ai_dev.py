#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:03:57 2023

@author: zc
"""

import classic.chess_rt as chess
import numpy as np


# This is for dev purpose.
if __name__ == "__main__":
    # create an environment
    env = chess.env('human')
    # reset the environment
    obs = env.reset()
    games = 1
    tick = 0
    TICK_COUNTDOWN = 200
    
    try:
        # AEC
        for agent in env.agent_iter():
            print("this is the agent", agent)
            env.unwrapped.set_color(agent)
            """
            TODO: Understand observation["observation"]
            
            Understanding the 8*8*(P+L) list maybe useful for the model training, 
            but the list is needed to confirm the values are correct, since it now
            records by pieces instead of only 2 players the original intention.
            """
            observation, reward, termination, truncation, info = env.last()
            
            # Only terminate when one side has no piece left
            if termination:
                print("Game Over")
                # Game counter
                games -= 1
                if games > 0:
                    obs = env.reset()
                    continue
                break
            # A piece is captured, it's truncated from alive agents
            if truncation:
                env.step(None)
                continue
            
            """
            TODO: Cool down mechanism
            
            Could use the api of KungFu chess or implement it ourself.
            """
            # Pieces only be ready to move when its status is stale
            if env.unwrapped.is_piece_ready(agent):
                # Filter out invalid actions
                valid_actions = np.where(np.array(observation["action_mask"]) == 1)[0]
                """
                TODO: Model training
                
                Machine learning part should replace the following line to choose an 
                action itself.
                """
                # Random choose a valid action
                action = np.random.choice(valid_actions)
                col = action//(5*41)
                row = (action//41)%5
                print("action index:", action)
                # The coordination is a subjective perspective to the player side, 
                # black side's position is mirrored vertically.
                print("move (x = {}, y = {}, c = {})".format(col, row, action-(col*5+row)*41))
                env.step(action)
            else:
                print("code:", env.unwrapped.code_of_passing, ", the piece is not ready.")
                env.step(env.unwrapped.code_of_passing)
            
            # print("reward:", reward)
    
            if env.unwrapped.render_mode != "human":
                print(env.render())
            print("tick:", tick)
            print()
        
            tick += 1
            if tick == TICK_COUNTDOWN:
                break
    
        env.close()
   
    except KeyboardInterrupt:
        env.close()
    
    # Parallel
    # while env.agents:
    #     actions = {}
    #     for agent in env.agents:
    #         """
    #         TODO: Cool down mechanism
            
    #         Could use the api of KungFu chess or implement it ourself.
    #         """
    #         # Filter out invalid actions
    #         valid_actions = np.where(np.array(obs[agent]["action_mask"]) == 1)[0]
    #         """
    #         TODO: Model training
            
    #         Machine learning part should replace the following line to choose an 
    #         action itself.
    #         """
    #         # Random choose a valid action
    #         action = np.random.choice(valid_actions)
            
    #         col = action//(8*74)
    #         row = (action//74)%8
    #         print(agent, "action index:", action)
    #         # The coordination is a subjective perspective to the player side, 
    #         # black side's position is mirrored vertically.
    #         print("move (x = {}, y = {}, c = {})".format(col, row, action-(col*8+row)*74))
            
    #         actions[agent] = action
        
        
    #     observations, rewards, terminations, truncations, infos = env.step(actions)
    #     obs = observations
        
    #     print(env.render())
    #     print()
        
    #     count_down -= 1
    #     if count_down == 0:
    #         break