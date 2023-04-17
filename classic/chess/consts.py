GARDNER_BOARD = [
        "rnbqk",
        "ppppp",
        ".....",
        "PPPPP",
        "RNBQK",
]
BOARD_ROW = 5
BOARD_COL = 5

CHESS_PIECES = ("R0", "N0", "B0", "Q", "K", "B1", "N1", "R1", "P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
MINI_CHESS_PIECES = ("R", "N", "B", "Q", "K", "P0", "P1", "P2", "P3", "P4")

BOARD = GARDNER_BOARD
BOARD_PIECES = MINI_CHESS_PIECES

TOTAL_MOVES = ((BOARD_COL-1) * 8 + 1) + 8 # QUEEN_MOVES + KNIGHT_MOVES
MOVE_TIME = 4 # Lowest time needed is 1
COOLDOWN_TIME = 4 # 0 is valid for no cooldown time
