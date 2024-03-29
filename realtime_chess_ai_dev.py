#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 14:03:57 2023

@author: zc
"""

import classic.chess_rt as chess
import numpy as np
import supersuit as ss

GAME_COUNTDOWN = 1
TICK_COUNTDOWN = 300

def aec(env):
    env.reset()
    tick = 0

    for agent in env.agent_iter():
        print("this is the agent", agent)
        observation, reward, termination, truncation, info = env.last()
        
        # Game over
        if all(env.terminations.values()):
            break
        # A piece is captured, it's terminated/truncated from alive agents
        if termination or truncation:
            env.step(None)
            continue
        
        # Filter out invalid actions
        valid_actions = np.where(np.array(observation["action_mask"]) == 1)[0]
        
        # Random choose a valid action
        action = np.random.choice(valid_actions)
        col = (action//41)%5
        row = action//(5*41)
        print("action index:", action)
        # The coordination is a subjective perspective to the player side, 
        # black side's position is mirrored vertically.
        print("move (x = {}, y = {}, c = {})".format(col, row, action-(row*5+col)*41))
        env.step(action)
        
        print("reward:", reward)

        env.render()
        print("tick:", tick)
        print()
    
        tick += 1
        if tick == TICK_COUNTDOWN:
            break

def parellel(env):
    obs, infos = env.reset(return_info=True)
    tick = 0

    while env.agents:
        actions = {}
        for agent in env.agents:
            print(agent)
            # Filter out invalid actions
            valid_actions = np.where(np.array(infos[agent]["action_mask"]) == 1)[0]

            action = None
            if len(valid_actions) > 0:
                # Random choose a valid action
                action = np.random.choice(valid_actions)
                col = (action//41)%5
                row = action//(5*41)
                print("action index:", action)
                # The coordination is a subjective perspective to the player side, 
                # black side's position is mirrored vertically.
                print("move (x = {}, y = {}, c = {})".format(col, row, action-(row*5+col)*41))
            
            actions[agent] = action
        
        observations, rewards, terminations, truncations, infos = env.step(actions)
        
        if all(terminations.values()):
            break

        obs = observations
        
        env.render()
        print("tick:", tick)
        print()
        
        tick += 1
        if tick == TICK_COUNTDOWN:
            break


# This is for dev purpose.
if __name__ == "__main__":
    # create an environment
    is_parellel = False
    env = chess.env(is_parellel=is_parellel, render_mode='human')
    # Preprocessing
    # env = ss.color_reduction_v0(env, mode="full")
    env = ss.dtype_v0(env, "float32")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    game = 0
    
    try:
        while game < GAME_COUNTDOWN:
            if is_parellel:
                parellel(env)
            else:
                aec(env)
            print("Game", game, "Over")
            game += 1
        
        env.close()
    except KeyboardInterrupt:
        env.close()