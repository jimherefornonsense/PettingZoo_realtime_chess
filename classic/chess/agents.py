# -*- coding: utf-8 -*-
from .consts import *
from enum import Enum
from collections import deque

class Status(Enum):
    IDLE = 0
    MOVING = 1
    COOLING = 2

class Agents:
    def __init__(self):
        self.tick = 0
        # an agent order list, agents could be deleted by the env when truncating them
        self._agent_order = []
        # agent-position dict, positions are subjective to agent side
        self._agent_position = {}
        # position-agent dict
        self._position_agent = {}
        
        self.init_agents()

    def init_agents(self):
        for i in range(BOARD_COL * BOARD_ROW):
            self._position_agent[i] = deque()
        
        # White and black pieces interchange
        for i, piece in enumerate(BOARD_PIECES):
            w_piece = "W_"+piece
            b_piece = "B_"+piece

            self._agent_order.append(w_piece)
            self._agent_order.append(b_piece)
            
            self._agent_position[w_piece] = {"pos": i, "status": Status.IDLE, "cooldown_end": 0}
            self._agent_position[b_piece] = {"pos": i, "status": Status.IDLE, "cooldown_end": 0}

            self._position_agent[i].append(w_piece)
            self._position_agent[self._mirror_pos(i)].append(b_piece)

    def get_list(self):
        return self._agent_order

    def reset(self):
        self.tick = 0
        self._agent_order.clear()
        self._agent_position.clear()
        self._position_agent.clear()

        self.init_agents()

    def _mirror_pos(self, sub_pos):
        # return sub_pos^0x38
        return (BOARD_ROW - sub_pos // BOARD_COL - 1) * BOARD_COL + sub_pos % BOARD_COL

    def get_status(self, agent):
        if (self._agent_position[agent]["status"] == Status.COOLING 
            and self._agent_position[agent]["cooldown_end"] <= self.tick):
            self.set_idle(agent)

        return self._agent_position[agent]["status"]

    def set_idle(self, agent):
        self._agent_position[agent]["status"] = Status.IDLE
    
    def set_moving(self, agent):
        self._agent_position[agent]["status"] = Status.MOVING

    def set_cooldown(self, agent):
        self._agent_position[agent]["status"] = Status.COOLING
        self._agent_position[agent]["cooldown_end"] = self.tick + COOLDOWN_TIME

    def set_next_pos(self, agent, next_pos):
        if self._agent_position[agent]["pos"] != next_pos:
            self.set_moving(agent)
        self._agent_position[agent]["pos"] = next_pos

    def get_pos(self, agent):
        if agent not in self._agent_position or self.get_status(agent) == Status.MOVING:
            return None
        return self._agent_position[agent]["pos"]

    def update_time(self):
        self.tick += 1

    def update_position(self, from_pos, to_pos, piece):
        """Return the captured piece (agent) or None if no piece is captured"""
        if from_pos == to_pos:
            return None
        if piece.islower():
            from_pos = self._mirror_pos(from_pos)
            to_pos = self._mirror_pos(to_pos)
        
        # Take up the piece
        agent = self._position_agent[from_pos].popleft()

        # Check target square
        captured_piece = None
        if len(self._position_agent[to_pos]) != 0:
            captured_piece = self._position_agent[to_pos][0]
            # Only remove the piece if the piece isn't moving
            if self.get_status(captured_piece) != Status.MOVING:
                self._agent_position.pop(captured_piece)
                self._position_agent[to_pos].popleft()
            else:
                captured_piece = None
                
        # Put down the piece
        self._position_agent[to_pos].append(agent)
        self.set_cooldown(agent)
        
        return captured_piece

    def find_last_alive(self, removed_agent):
        i = self._agent_order.index(removed_agent)
        # self._agent_order.remove(removed_agent)
        
        return self._agent_order[i-1]

    def generate_agent_map(self):
        moving_agent = []
        
        for agent, data in self._agent_position.items():
            pos = data["pos"]
            if agent[:1] == "B":
                pos = self._mirror_pos(pos)
            
            x = pos % BOARD_COL
            y = pos // BOARD_COL
            
            if data["status"] == Status.MOVING:
                moving_agent.append((x, y, agent))
            else:
                yield x, y, agent
                
        # Place moving pieces in the end to ensure their image layers are higher
        for x, y, agent in moving_agent:
            yield x, y, agent
        