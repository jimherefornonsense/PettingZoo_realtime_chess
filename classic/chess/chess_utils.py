import numpy as np
from .consts import *

class Move:
    def __init__(self, uci, piece):
        self.uci = uci
        self.from_square = (int(uci[1]) - 1) * BOARD_COL + (ord(uci[0]) - ord("a"))
        self.to_square = (int(uci[3]) - 1) * BOARD_COL + (ord(uci[2]) - ord("a"))
        self.piece = piece
        self.promotion = True if self.piece.lower() == "p" and int(uci[3]) == BOARD_ROW else False
    
    def __str__(self) -> str:
        return self.uci


def boards_to_ndarray(boards):
    arr64 = np.array(boards, dtype=np.uint64)
    arr8 = arr64.view(dtype=np.uint8)
    bits = np.unpackbits(arr8)
    floats = bits.astype(bool)
    boardstack = floats.reshape([len(boards), 8, 8])
    boardimage = np.transpose(boardstack, [1, 2, 0])
    return boardimage


def square_to_coord(s):
    col = s % BOARD_COL
    row = s // BOARD_COL
    return (col, row)


def diff(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return (x2 - x1, y2 - y1)


def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)


def get_queen_dir(diff):
    dx, dy = diff
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    magnitude = max(max(abs(dx), abs(dy)) - 1, 0)

    assert magnitude < BOARD_COL and magnitude >= 0
    counter = 0
    for x in range(-1, 1 + 1):
        for y in range(-1, 1 + 1):
            if magnitude != 0 and (x == 0 and y == 0):
                continue
            if x == sign(dx) and y == sign(dy):
                return magnitude, counter
            counter += 1
    assert False, "bad queen move inputted"


def get_queen_plane(diff):
    NUM_COUNTERS = 8
    IDLE_COUNTED = 9
    mag, counter = get_queen_dir(diff)

    if mag == 0:
        return counter
    return (mag - 1) * NUM_COUNTERS + IDLE_COUNTED + counter


def get_knight_dir(diff):
    dx, dy = diff
    counter = 0
    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y:
                    return counter
                counter += 1
    assert False, "bad knight move inputted"


def is_knight_move(diff):
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2


def get_pawn_promotion_move(diff):
    dx, dy = diff
    assert dy == 1
    assert -1 <= dx <= 1
    return dx + 1


def get_pawn_promotion_num(promotion):
    assert (
        promotion == chess.KNIGHT
        or promotion == chess.BISHOP
        or promotion == chess.ROOK
    )
    return 0 if promotion == chess.KNIGHT else (1 if promotion == chess.BISHOP else 2)


def move_to_coord(move):
    return square_to_coord(move.from_square)


def get_move_plane(move):
    source = move.from_square
    dest = move.to_square
    difference = diff(square_to_coord(source), square_to_coord(dest))

    QUEEN_MOVES = 8 * (BOARD_COL-1) + 1 # add non-moving action
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference):
        return KNIGHT_OFFSET + get_knight_dir(difference)
    else:
        # if move.promotion is not None and move.promotion != chess.QUEEN:
        #     return (
        #         UNDER_OFFSET
        #         + 3 * get_pawn_promotion_move(difference)
        #         + get_pawn_promotion_num(move.promotion)
        #     )
        # else:
        return QUEEN_OFFSET + get_queen_plane(difference)


moves_to_actions = {}
actions_to_moves = {}


def action_to_move(board, action):
    uci = actions_to_moves[action]
    move = Move(uci, board.piece_at(uci[:2]))
    
    return move


def make_move_mapping(uci_move, board):
    move = Move(uci_move, board.piece_at(uci_move[:2]))
    source = move.from_square
    
    coord = square_to_coord(source)
    panel = get_move_plane(move)
    action = (coord[1] * BOARD_COL + coord[0]) * TOTAL_MOVES + panel

    moves_to_actions[uci_move] = action
    actions_to_moves[action] = uci_move


def legal_moves(board, agent_pos = None):
    """Returns legal moves.

    action space is a 5x5x74 dimensional array
    Each of the 5×5
    positions identifies the square from which to “pick up” a piece. The first 57 planes encode
    possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be
    moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The
    next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible
    underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or
    rook respectively. Other pawn moves or captures from the seventh rank are promoted to a
    queen
    """
    legal_moves = []
    
    for move in board.generate_all_moves():
        if agent_pos == None:
            if move not in moves_to_actions:
                make_move_mapping(move, board)
            legal_moves.append(moves_to_actions[move])
        else:
            if move not in moves_to_actions:
                make_move_mapping(move, board)
            move = Move(move, board.piece_at(move[:2]))
            if move.from_square == agent_pos: # Mapping the current piece
                legal_moves.append(moves_to_actions[move.uci])
    
    return legal_moves