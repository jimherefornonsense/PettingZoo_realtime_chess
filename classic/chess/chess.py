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

The observation is a dictionary which contains an `'observation'` element which is the usual RL observation described below, and an  `'action_mask'` which holds the legal moves, described in the Legal Actions Mask section.

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

We instead flatten this into 5×5×74 = 1850 discrete action space.

You can get back the original (x,y,c) coordinates from the integer action `a` with the following expression: `(a/(5*74), (a/74)%5, a-(x*5+y)*74`

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
from os import path

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, aec_to_parallel
from pettingzoo.utils.agent_selector import agent_selector

from . import chess_utils

from .mini_chess.mini_chess import MiniChess
from .mini_chess.const import *


def env(render_mode=None):
    env = raw_env(render_mode=render_mode)
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    # env = aec_to_parallel(env)
    return env


class raw_env(AECEnv):

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "name": "chess_rt",
        "is_parallelizable": True,
        "render_fps": 2,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        self.board = MiniChess(GARDNER_BOARD)

        self.agents = chess_utils.generate_agents()
        self.possible_agents = self.agents[:]

        self._agent_selector = agent_selector(self.agents)

        self.action_spaces = {name: spaces.Discrete(BOARD_COL * BOARD_ROW * 74) for name in self.agents}
        self.observation_spaces = {
            name: spaces.Dict(
                {
                    "observation": spaces.Box(
                        low=0, high=1, shape=(BOARD_COL, BOARD_ROW, 111), dtype=bool
                    ),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(BOARD_COL * BOARD_ROW * 74,), dtype=np.int8
                    ),
                }
            )
            for name in self.agents
        }

        self.rewards = None
        self.infos = {name: {} for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.terminations = {name: False for name in self.agents}

        self.agent_selection = None
        self._last_alive_agent = None
        self._dead_step_initializer = None
        
        self.board_history = np.zeros((BOARD_COL, BOARD_ROW, 104), dtype=bool)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        if self.render_mode in {"human", "rgb_array"}:
            try:
                import pygame
            except ImportError:
                raise DependencyNotInstalled(
                    f"pygame is needed for {self.render_mode} rendering, run with `pip install pettingzoo[classic]`"
                )

            self.BOARD_SIZE = (400, 400)
            self.window_surface = None
            self.clock = pygame.time.Clock()
            self.cell_size = (self.BOARD_SIZE[0] / BOARD_COL, self.BOARD_SIZE[1] / BOARD_ROW)

            bg_name = path.join(path.dirname(__file__), "img/chessboard.png")
            self.bg_image = pygame.transform.scale(
                pygame.image.load(bg_name), (self.BOARD_SIZE[0]+240, self.BOARD_SIZE[1]+240)
            )

            def load_piece(file_name):
                img_path = path.join(path.dirname(__file__), f"img/{file_name}.png")
                return pygame.transform.scale(
                    pygame.image.load(img_path), self.cell_size
                )

            self.piece_images = {
                "p": [load_piece("pawn_white"), load_piece("pawn_black")],
                "n": [load_piece("knight_white"), load_piece("knight_black")],
                "b": [load_piece("bishop_white"), load_piece("bishop_black")],
                "r": [load_piece("rook_white"), load_piece("rook_black")],
                "q": [load_piece("queen_white"), load_piece("queen_black")],
                "k": [load_piece("king_white"), load_piece("king_black")],
            }

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
    
    def set_color(self, agent):
        self.board.cur_color("w" if agent[:1] == 'W' else "b")

    def observe(self, agent):
        self.set_color(agent)
        
        # observation = chess_utils.get_observation(
        #     self.board, 1 if agent[:1] == 'B' else 0
        # )
        # observation = np.dstack((observation[:, :, :7], self.board_history))
        
        legal_moves = chess_utils.legal_moves(self.board, agent)

        action_mask = np.zeros(BOARD_COL * BOARD_ROW * 74, "int8")
        
        for i in legal_moves:
            action_mask[i] = 1

        return {"observation": None, "action_mask": action_mask}

    def reset(self, seed=None, return_info=False, options=None):
        self.has_reset = True

        self.agents = self.possible_agents[:]
        
        chess_utils.reset_agent_table()

        self.board = MiniChess(GARDNER_BOARD)

        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {name: 0 for name in self.agents}
        self._cumulative_rewards = {name: 0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.board_history = np.zeros((8, 8, 104), dtype=bool)

        if self.render_mode == "human":
            self.render()
            
    def _reset_next_agent(self):
        agent_idx = self.agents.index(self._last_alive_agent)
        self._agent_selector._current_agent = agent_idx+1
    
    def _sync_next_agent(self):
        self._reset_next_agent()
        self.agent_selection = self._agent_selector.next()

    def set_game_result(self, result_val):
        for i, name in enumerate(self.agents):
            self.terminations[name] = True
            result_coef = 1 if i == 0 else -1
            self.rewards[name] = result_val * result_coef
            self.infos[name] = {"legal_moves": []}

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
                
            print("Agent is dead")
            print()
            self._was_dead_step(action)
            
            if self._dead_step_initializer == self.agent_selection:
                self._sync_next_agent()
                self._dead_step_initializer = None
                
            return
        
        self._last_alive_agent = self.agent_selection
        current_agent = self.agent_selection
        chosen_move = chess_utils.action_to_move(self.board, action, current_agent)
        
        assert self.board.push(chosen_move.uci) != False
        
        captured_piece = chess_utils.update_position(current_agent, chosen_move)
        if captured_piece:
            self.truncations[captured_piece] = True

        self.board.cur_color("b" if self.agent_selection[:1] == 'W' else "w")
        next_legal_moves = chess_utils.legal_moves(self.board)
        
        if not any(next_legal_moves):
            result_val = chess_utils.result_to_int("1-0")
            self.set_game_result(result_val)
        
        # is_stale_or_checkmate = not any(next_legal_moves)
        

        # claim draw is set to be true to align with normal tournament rules

        # is_repetition = self.board.is_repetition(3)
        # is_50_move_rule = self.board.can_claim_fifty_moves()
        # is_claimable_draw = is_repetition or is_50_move_rule
        # game_over = is_claimable_draw or is_stale_or_checkmate

        """
        TODO: Reward structure
        
        When a piece captures a piece the other side, gives rewards.
        """
        # if game_over:
            # result = self.board.result(claim_draw=True)
            # result_val = chess_utils.result_to_int(result)
            # self.set_game_result(result_val)

        self._accumulate_rewards()

        # Update board after applying action
        # next_board = chess_utils.get_observation(self.board, current_agent)
        # self.board_history = np.dstack(
        #     (next_board[:, :, 7:], self.board_history[:, :, :-13])
        # )
        self.agent_selection = (
            self._agent_selector.next()
        )  # Give turn to the next agent
        
        if self.render_mode == "human":
            self.render()

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
        elif self.render_mode == "ansi":
            return str(self.board)
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_gui()
        else:
            raise ValueError(
                f"{self.render_mode} is not a valid render mode. Available modes are: {self.metadata['render_modes']}"
            )

    def _render_gui(self):
        try:
            import pygame
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install pettingzoo[classic]`"
            )

        if self.window_surface is None:
            pygame.init()

            if self.render_mode == "human":
                pygame.display.init()
                pygame.display.set_caption("Chess")
                self.window_surface = pygame.display.set_mode(self.BOARD_SIZE)
            elif self.render_mode == "rgb_array":
                self.window_surface = pygame.Surface(self.BOARD_SIZE)

        self.window_surface.blit(self.bg_image, (0, 0), (0,0, self.BOARD_SIZE[0], self.BOARD_SIZE[1]))
        for x, y, piece in self.board.piece_map():
            color = 0 if piece.isupper() else 1
            pos = (x * self.cell_size[0], y * self.cell_size[1])
            piece_img = self.piece_images[piece.lower()][color]
            self.window_surface.blit(piece_img, pos)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )

    def close(self):
        if self.render_mode == "human":
            print("bye!")
            try:
                import pygame
                import sys
            except ImportError:
                raise DependencyNotInstalled(
                    "pygame is not installed, run `pip install pettingzoo[classic]`"
                )
            pygame.quit()
            sys.exit()
    
# class parallel_env(ParallelEnv):
#     def __init__(self):
#         pass

#     def reset(self, seed=None, return_info=False, options=None):
#         pass
    
#     def step(self, actions):
#         pass
    
#     def render(self):
#         pass
    
#     def observation_space(self, agent):
#         return self.observation_spaces[agent]
    
#     def action_space(self, agent):
#         return self.action_spaces[agent]
