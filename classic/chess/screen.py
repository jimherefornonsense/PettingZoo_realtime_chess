# -*- coding: utf-8 -*-

import pygame
from os import path, listdir, remove
import sys
import numpy as np

class Screen:
    folder_path = path.join(path.dirname(__file__), "..", "..", "screenshots")
    
    def __init__(self, col, row, agent_map):
        self.row = row
        self.BOARD_SIZE = (400, 400)
        self.window_surface = None
        self.display_surface = None
        self.clock = pygame.time.Clock()
        self.cell_size = (self.BOARD_SIZE[0] / col, self.BOARD_SIZE[1] / row)
        self.frame_count = 0

        # Loading
        bg_name = path.join(path.dirname(__file__), "img/chessboard.png")
        scale_from_8x8 = (self.BOARD_SIZE[0] + (8-col)*self.cell_size[0], self.BOARD_SIZE[1] + (8-row)*self.cell_size[1])
        self.bg_image = pygame.transform.scale(pygame.image.load(bg_name), scale_from_8x8)

        def load_piece(file_name):
            img_path = path.join(path.dirname(__file__), f"img/{file_name}.png")
            return pygame.transform.scale(
                pygame.image.load(img_path), self.cell_size
            )

        self.piece_images = {
            "P": [load_piece("pawn_white"), load_piece("pawn_black")],
            "N": [load_piece("knight_white"), load_piece("knight_black")],
            "B": [load_piece("bishop_white"), load_piece("bishop_black")],
            "R": [load_piece("rook_white"), load_piece("rook_black")],
            "Q": [load_piece("queen_white"), load_piece("queen_black")],
            "K": [load_piece("king_white"), load_piece("king_black")],
        }
        
        # Delete all images in the folder
        file_list = listdir(self.folder_path)
        for filename in file_list:
            file_path = path.join(self.folder_path, filename)
            remove(file_path)
        
        # Pieces' positions
        self.agent_data = {}
        self.agent_next_pos = {}
        self.init_board(agent_map)
        
    def init_board(self, agent_map):
        for x, y, agent in agent_map:
            color = 0 if agent[:1] == "W" else 1
            piece = agent[2:3]
            coord = self.coord_map(x, y)
            piece_img = self.piece_images[piece][color]
            rect = piece_img.get_rect()
            rect.x = coord[0]
            rect.y = coord[1]
            
            self.agent_data[agent] = {"img": piece_img, "rect": rect, "dx": 0.0, "dy": 0.0}
            self.agent_next_pos[agent] = coord
    
    def reset(self, agent_map):
        self.agent_data.clear()
        self.agent_next_pos.clear()
        # self.frame_count = 0

        self.init_board(agent_map)
    
    def coord_map(self, x, y):
        y = self.row - y - 1 
        return (x * self.cell_size[0], y * self.cell_size[1])
        
    def update_pos(self, x, y, agent, move_time):
        coord = self.coord_map(x, y)
        if self.agent_next_pos[agent] != coord:
            self.agent_next_pos[agent] = coord
            self.update_delta_offset(agent, coord, move_time)
    
    def update_delta_offset(self, agent, next_coord, move_time):
        delta_x = (next_coord[0] - self.agent_data[agent]["rect"].x) / move_time
        delta_y = (next_coord[1] - self.agent_data[agent]["rect"].y) / move_time
        self.agent_data[agent]["dx"] = delta_x
        self.agent_data[agent]["dy"] = delta_y

    def update_frame(self, agent_map, move_time=None):
        if self.window_surface is None:
            pygame.init()
            self.window_surface = pygame.Surface(self.BOARD_SIZE)

        self.window_surface.blit(self.bg_image, (0, 0), (0,0, self.BOARD_SIZE[0], self.BOARD_SIZE[1]))
        
        for x, y, agent in agent_map:
            self.update_pos(x, y, agent, move_time)
            if abs(self.agent_next_pos[agent][0] - self.agent_data[agent]["rect"].x) < abs(self.agent_data[agent]["dx"]):
                self.agent_data[agent]["rect"].x = self.agent_next_pos[agent][0]
                self.agent_data[agent]["dx"] = 0
                
            if abs(self.agent_next_pos[agent][1] - self.agent_data[agent]["rect"].y) < abs(self.agent_data[agent]["dy"]):
                self.agent_data[agent]["rect"].y = self.agent_next_pos[agent][1]
                self.agent_data[agent]["dy"] = 0
                # Pawn promotion
                if agent[2:3] == "P" and (y == 0 or y == self.row - 1):
                    color = 0 if agent[:1] == "W" else 1
                    self.agent_data[agent]["img"] = self.piece_images["Q"][color]
            
            dx = self.agent_data[agent]["dx"]
            dy = self.agent_data[agent]["dy"]
            self.agent_data[agent]["rect"].move_ip(dx, dy)
                
            self.window_surface.blit(self.agent_data[agent]["img"], self.agent_data[agent]["rect"])

    def render(self, mode):
        if mode == "human":
            if not self.display_surface:
                pygame.display.init()
                pygame.display.set_caption("Chess")
                self.display_surface = pygame.display.set_mode(self.BOARD_SIZE)

            self.display_surface.blit(self.window_surface, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            # Capture the current screen and save it as a PNG file
            pygame.image.save(self.display_surface, f"{self.folder_path}/frame_{self.frame_count:03}.png")
            self.frame_count += 1
        elif mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )
    
    def close(self, mode):
        print("bye!")
        pygame.quit()
        if mode == "human":
            sys.exit()
        
if __name__ == "__main__":
    # initialize pygame
    pygame.init()
    
    # create the game window
    screen = pygame.display.set_mode((800, 600))
    
    # load the image and create the rect object
    image = pygame.image.load('img/pawn_white.png')
    rect = image.get_rect()
    
    # set the initial position of the rect
    rect.x = 0
    rect.y = 300
    
    # set the speed of the rect
    speed = 5
    
    # game loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # update the position of the rect
        rect.x += speed
        
        # draw the image to the screen
        screen.blit(image, rect)
        
        # update the display
        pygame.display.update()
    
    # quit pygame
    pygame.quit()