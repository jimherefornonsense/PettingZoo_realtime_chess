# -*- coding: utf-8 -*-

CHESS_PIECES = ("R0", "N0", "B0", "Q", "K", "B1", "N1", "R1", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")

# agent-position dict, positions are subjective to agent side
agent_position = {}
# position-agent dict, positions are 0-63 indices to the board's square
position_agent = {i: None for i in range(64)}

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
    for i, piece in enumerate(CHESS_PIECES):
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
    position_agent = {i: None for i in range(64)}
    generate_agents()

def _mirror_pos(sub_pos):
    return sub_pos^0x38
    
def update_position(agent, from_pos, to_pos):
    if from_pos == to_pos:
        return None
    
    captured_piece = position_agent[to_pos]
    
    position_agent[from_pos] = None    
    position_agent[to_pos] = agent
    
    if agent[:1] == 'B':
        to_pos = _mirror_pos(to_pos)
    agent_position[agent] = to_pos
    
    return captured_piece
