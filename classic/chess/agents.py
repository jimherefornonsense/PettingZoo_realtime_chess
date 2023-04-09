# -*- coding: utf-8 -*-
from .mini_chess.const import *

CHESS_PIECES = ("R0", "N0", "B0", "Q", "K", "B1", "N1", "R1", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
MINI_CHESS_PIECES = ("R", "N", "B", "Q", "K", "P0", "P1", "P2", "P3", "P4")

# agent-position dict, positions are subjective to agent side
agent_position = {}
# position-agent dict
position_agent = {i: None for i in range(BOARD_COL * BOARD_ROW)}

def generate_agents():
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
        
        agent_position[w_piece] = i
        agent_position[b_piece] = i
        
        position_agent[i] = w_piece
        position_agent[_mirror_pos(i)] = b_piece
    
    return agents

def reset():
    agent_position = {}
    # position_agent = {i: None for i in range(64)}
    position_agent = {i: None for i in range(BOARD_COL * BOARD_ROW)}
    generate_agents()

def _mirror_pos(sub_pos):
    # return sub_pos^0x38
    return (BOARD_ROW - sub_pos // BOARD_COL - 1) * BOARD_COL + sub_pos % BOARD_COL
    
def update_position(agent, from_pos, to_pos):
    if from_pos == to_pos:
        return None
    
    agent_position[agent] = to_pos
    
    if agent[:1] == 'B':
        from_pos = _mirror_pos(from_pos)
        to_pos = _mirror_pos(to_pos)
    
    captured_piece = position_agent[to_pos]
    position_agent[from_pos] = None    
    position_agent[to_pos] = agent
    
    return captured_piece
