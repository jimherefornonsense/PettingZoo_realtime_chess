# noqa
"""
# Chess

```{figure} classic_chess.gif
:width: 140px
:name: chess
```

This environment is part of the <a href='..'>classic environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.classic.chess_v5` |
|--------------------|------------------------------------|
| Actions            | Discrete                           |
| Parallel API       | Yes                                |
| Manual Control     | No                                 |
| Agents             | `agents= ['player_0', 'player_1']` |
| Agents             | 2                                  |
| Action Shape       | Discrete(4672)                     |
| Action Values      | Discrete(4672)                     |
| Observation Shape  | (8,8,20)                           |
| Observation Values | [0,1]                              |


Chess is one of the oldest studied games in AI. Our implementation of the observation and action spaces for chess are what the AlphaZero method uses, with two small changes.

### Observation Space

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

Like AlphaZero, the main observation space is an 8x8 image representing the board. It has 20 channels representing:

* Channels 0 - 3: Castling rights:
  * Channel 0: All ones if white can castle queenside
  * Channel 1: All ones if white can castle kingside
  * Channel 2: All ones if black can castle queenside
  * Channel 3: All ones if black can castle kingside
* Channel 4: Is black or white
* Channel 5: A move clock counting up to the 50 move rule. Represented by a single channel where the *n* th element in the flattened channel is set if there has been *n* moves
* Channel 6: All ones to help neural networks find board edges in padded convolutions
* Channel 7 - 18: One channel for each piece type and player color combination. For example, there is a specific channel that represents black knights. An index of this channel is set to 1 if a black knight is in the corresponding spot on the game board, otherwise, it is set to 0. En passant
possibilities are represented by displaying the vulnerable pawn on the 8th row instead of the 5th.
* Channel 19: represents whether a position has been seen before (whether a position is a 2-fold repetition)

Like AlphaZero, the board is always oriented towards the current agent (the currant agent's king starts on the 1st row). In other words, the two players are looking at mirror images of the board, not the same board.

Unlike AlphaZero, the observation space does not stack the observations previous moves by default. This can be accomplished using the `frame_stacking` argument of our wrapper.

#### Legal Actions Mask

The legal moves available to the current agent are found in the `action_mask` element of the dictionary observation. The `action_mask` is a binary vector where each index of the vector represents whether the action is legal or not. The `action_mask` will be all zeros for any agent except the one
whose turn it is. Taking an illegal move ends the game with a reward of -1 for the illegally moving agent and a reward of 0 for all other agents.

### Action Space

From the AlphaZero chess paper:

> [In AlphaChessZero, the] action space is a 8x8x73 dimensional array.
Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 57 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
queen.

We instead flatten this into 5×5×41 = 1025 discrete action space.

You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a/41)%5, (a/(5*41), a-(y*5+x)*41`

### Rewards

| Winner | Loser | Draw |
| :----: | :---: | :---: |
| +1     | -1    | 0 |

### Version History

* rt: Realtime mini chess env
* v5: Changed python-chess version to version 1.7 (1.13.1)
* v4: Changed observation space to proper AlphaZero style frame stacking (1.11.0)
* v3: Fixed bug in arbitrary calls to observe() (1.8.0)
* v2: Legal action mask in observation replaced illegal move list in infos (1.5.0)
* v1: Bumped version of all environments due to adoption of new agent iteration scheme where all agents are iterated over after they are done (1.4.0)
* v0: Initial versions release (1.0.0)

"""
import gymnasium
import numpy as np
from gymnasium import spaces

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import agent_selector

from . import chess_utils
from . import agents
from . import screen

from mini_chess.mini_chess import MiniChess
from .consts import *

def env(is_parellel = False, render_mode = None):
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    if is_parellel:
        env = parellel_wrapper(env)
    return env


class raw_env(AECEnv):

    metadata = {
        "render_modes": ["ansi", "human", "rgb_array"],
        "name": "chess_rt",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.board = MiniChess(BOARD[:], MOVE_TIME)

        self.agent_table = agents.Agents()
        self.agents = self.agent_table.get_list()
        self.possible_agents = self.agents[:]
        self._agent_selector = agent_selector(self.agents)
        self._last_alive_agent = None
        self._dead_step_initializer = None

        self.screen = screen.Screen(BOARD_COL, BOARD_ROW, self.agent_table.generate_agent_map())
        
        # Codes for all actions and the last code for passing the round
        self.code_of_passing = BOARD_COL * BOARD_ROW * TOTAL_MOVES
        self.action_spaces = {name: spaces.Discrete(BOARD_COL * BOARD_ROW * TOTAL_MOVES + 1) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=255, shape=(self.screen.BOARD_SIZE[0], self.screen.BOARD_SIZE[1], 3), dtype=np.uint8,),
                    "action_mask": spaces.Box(low=0, high=1, shape=(BOARD_COL * BOARD_ROW * TOTAL_MOVES + 1,), dtype=np.int8)
                }
            )
            for name in self.agents
        }
        self.rewards = {name: 0 for name in self.agents}
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.agent_selection = None
        self.render_mode = render_mode
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def _set_color(self, agent):
        self.board.cur_color(agent[:1].lower())

    def observe(self, agent):
        self._set_color(agent)
        observation = self.screen.render("rgb_array")
        action_mask = np.zeros(BOARD_COL * BOARD_ROW * TOTAL_MOVES + 1, "int8")
        
        if not (self.terminations[agent] or self.truncations[agent]):
            if self._is_piece_ready(agent):
                legal_moves = chess_utils.legal_moves(self.board, self.agent_table.get_pos(agent))
                for i in legal_moves:
                    action_mask[i] = 1
            else: 
                action_mask[self.code_of_passing] = 1

        return {"observation": observation, "action_mask": action_mask} 

    def reset(self, seed=None, return_info=False, options=None):
        self.has_reset = True

        self.board = MiniChess(BOARD[:], MOVE_TIME)

        self.agent_table.reset()
        self.agents = self.agent_table.get_list()
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        self._last_alive_agent = None
        self._dead_step_initializer = None

        self.screen.reset(self.agent_table.generate_agent_map())
        self.screen.update_frame(self.agent_table.generate_agent_map(), MOVE_TIME)

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}
            
    def _is_piece_ready(self, agent):
        return True if self.agent_table.get_status(agent) == agents.Status.IDLE else False

    def _update_state(self):
        self.agent_table.update_time()
        exec_move = self.board.update_time()
        if exec_move != None:
            exec_move = chess_utils.Move(exec_move[0], exec_move[1])
            agent, captured_agent = self.agent_table.update_position(exec_move.from_square, exec_move.to_square, exec_move.piece)
            
            if captured_agent:
                self._reward_capturing(agent, captured_agent)

            if captured_agent != self.agent_selection:
                self._last_alive_agent = self.agent_selection
        else:
            self._last_alive_agent = self.agent_selection
        
        self.screen.update_frame(self.agent_table.generate_agent_map(), MOVE_TIME)

    def _reward_capturing(self, agent, captured_agent):
        coef = 1 if agent[:1] != captured_agent[:1] else -1
        captured_piece = captured_agent[2:3]
        reward = 0

        if captured_piece == "P":
            reward = 1            
        elif captured_piece == "N":
            reward = 3
        elif captured_piece == "R" or captured_piece == "B":
            reward = 5
        elif captured_piece == "Q":
            reward = 7
        elif captured_piece == "K":
            reward = 9

        self.rewards[agent] = reward * coef
        self.terminations[captured_agent] = True

    def _reward_winning(self, color):
        for agent in self.agents:
            self.terminations[agent] = True
            if agent[:1] == color:
                self.rewards[agent] = 10

    def _reset_next_agent(self):
        agent_idx = self.agents.index(self._last_alive_agent)
        self._agent_selector._current_agent = agent_idx+1

    def _sync_next_agent(self):
        self._reset_next_agent()
        self.agent_selection = self._agent_selector.next()

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # In order to fit the mechanism of the env, which will gather all dead 
            # agents and delete them each but then it still loops back to the first 
            # deleted dead agent and causes KeyError, the checking point is to 
            # ensure to skip the deleted agent and loop for the next alive agent.
            #
            # It's a hecky way to make the env to fit the need of the realtime 
            # chess without modifying the source code.
            if not self._dead_step_initializer:
                self._dead_step_initializer = self.agent_selection

            # print("Agent is dead")
            # print()
            self._was_dead_step(action)

            if self._dead_step_initializer == self.agent_selection:
                self._sync_next_agent()
                self._dead_step_initializer = None

            return
        
        if action != self.code_of_passing:
            chosen_move = chess_utils.action_to_move(self.board, action)
            assert self.board.push(chosen_move.uci) != False
            self.agent_table.set_next_pos(self.agent_selection, chosen_move.to_square)
        
        # Update board to next unit time
        self._update_state()

        # self.board.cur_color("b" if self.agent_selection[:1] == 'W' else "w")
        # next_legal_moves = chess_utils.legal_moves(self.board)
        
        winner = self.board.has_winner()
        if winner != None:
            self._reward_winning("W" if winner == 0 else "B")
        
        # is_stale_or_checkmate = not any(next_legal_moves)

        # claim draw is set to be true to align with normal tournament rules

        # is_repetition = self.board.is_repetition(3)
        # is_50_move_rule = self.board.can_claim_fifty_moves()
        # is_claimable_draw = is_repetition or is_50_move_rule
        # game_over = is_claimable_draw or is_stale_or_checkmate

        self._accumulate_rewards()
        # Reset the reward table for the next step
        self.rewards = {name: 0 for name in self.agents}

        self.agent_selection = (
            self._agent_selector.next()
        )  # Give turn to the next agent

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self.screen.render(self.render_mode)
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def close(self):
        if self.screen:
            self.screen.close(self.render_mode)

from collections import defaultdict
from pettingzoo.utils.conversions import aec_to_parallel_wrapper

class parellel_wrapper(aec_to_parallel_wrapper):
    def step(self, actions):
        rewards = defaultdict(int)
        terminations = {}
        truncations = {}
        infos = {}
        observations = {}
        visited = set()

        self.aec_env._has_updated = True
        for agent in self.aec_env.agent_iter():
            observation, reward, termination, truncation, info = self.aec_env.last()
            terminations[agent] = termination
            truncations[agent] = truncation
            infos[agent] = info
            
            if termination or truncation:
                self.aec_env.step(None)
                visited.add(agent)
                continue

            if agent in visited:
                break
            visited.add(agent)

            # Check if action is valid still
            if not observation["action_mask"][actions[agent]]:
                valid_actions = np.where(np.array(observation["action_mask"]) == 1)[0]
                for action in valid_actions:
                    col = (action//41)%5
                    row = action//(5*41)
                    c = action-(row*5+col)*41
                    if c == 4:
                        actions[agent] = action
                        break

            self.aec_env.step(actions[agent])
            
            for agent in self.aec_env.agents:
                rewards[agent] += self.aec_env.rewards[agent]

        observations = {
            agent: self.aec_env.observe(agent) for agent in self.aec_env.agents
        }

        self.agents = self.aec_env.agents
        return observations, rewards, terminations, truncations, infos