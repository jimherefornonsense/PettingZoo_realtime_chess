# -*- coding: utf-8 -*-
from .mini_chess.const import *
from enum import Enum
from collections import deque

CHESS_PIECES = ("R0", "N0", "B0", "Q", "K", "B1", "N1", "R1", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
MINI_CHESS_PIECES = ("R", "N", "B", "Q", "K", "P0", "P1", "P2", "P3", "P4")

class Status(Enum):
    IDLE = 0
    MOVING = 1
    COOLING = 2

agent_order = []
# agent-position dict, positions are subjective to agent side
agent_position = {}
# position-agent dict
position_agent = {i: deque() for i in range(BOARD_COL * BOARD_ROW)}

def init_agents():
    agents = []

    # All white pieces go first
    # for i, piece in enumerate(CHESS_PIECES):
    #     w_piece = "W_"+piece
    #     agents.append(w_piece)
    #     agent_position[w_piece] = i
    #     position_agent[i] = w_piece
        
    # for i, piece in enumerate(CHESS_PIECES):
    #     b_piece = "B_"+piece
    #     agents.append(b_piece)
    #     agent_position[b_piece] = i
    #     position_agent[_mirror_pos(i)] = b_piece
    
    # White and black pieces interchange
    for i, piece in enumerate(MINI_CHESS_PIECES):        
        w_piece = "W_"+piece
        b_piece = "B_"+piece
        agents.append(w_piece)
        agents.append(b_piece)
        agent_order.append(w_piece)
        agent_order.append(b_piece)
        
        agent_position[w_piece] = {"pos": i, "status": Status.IDLE}
        agent_position[b_piece] = {"pos": i, "status": Status.IDLE}

        position_agent[i].clear()
        position_agent[_mirror_pos(i)].clear()
        position_agent[i].append(w_piece)
        position_agent[_mirror_pos(i)].append(b_piece)

    return agents

def reset():
    agent_position = {}
    init_agents()

def _mirror_pos(sub_pos):
    # return sub_pos^0x38
    return (BOARD_ROW - sub_pos // BOARD_COL - 1) * BOARD_COL + sub_pos % BOARD_COL

def set_status(agent, status):
    agent_position[agent]["status"] = status

def get_status(agent):
    return agent_position[agent]["status"]

def set_next_pos(agent, next_pos):
    if agent_position[agent]["pos"] != next_pos:
        agent_position[agent]["status"] = Status.MOVING
    agent_position[agent]["pos"] = next_pos
    
def get_pos(agent):
    if agent not in agent_position or agent_position[agent]["status"] != Status.IDLE:
        return None
    return agent_position[agent]["pos"]

def update_position(from_pos, to_pos, piece):
    if from_pos == to_pos:
        return None
    if piece.islower():
        from_pos = _mirror_pos(from_pos)
        to_pos = _mirror_pos(to_pos)
    
    # Take up the piece
    agent = position_agent[from_pos].popleft()

    # Check target square
    captured_piece = None
    if len(position_agent[to_pos]) != 0:
        captured_piece = position_agent[to_pos][0]
        # Only remove the piece if the piece isn't moving
        if get_status(captured_piece) != Status.MOVING:
            agent_position.pop(captured_piece)
            position_agent[to_pos].popleft()
            
    # Put down the piece
    position_agent[to_pos].append(agent)
    agent_position[agent]["status"] = Status.IDLE
    
    return captured_piece

def find_last_alive(removed_agent):
    i = agent_order.index(removed_agent)
    agent_order.remove(removed_agent)
    
    return agent_order[i-1]

def generate_agent_map():
    moving_agent = []
    
    for agent, data in agent_position.items():
        pos = data["pos"]
        if agent[:1] == "B":
            pos = _mirror_pos(pos)
        
        x = pos % BOARD_COL
        y = pos // BOARD_COL
        
        if data["status"] == Status.MOVING:
            moving_agent.append((x, y, agent))
        else:
            yield x, y, agent
            
    # Place moving pieces in the end to ensure their image layers are higher
    for x, y, agent in moving_agent:
        yield x, y, agent
    