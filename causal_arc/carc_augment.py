# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import math

# Custom modules.
from carc_utils import UtilsARC


class AugmentARC():

    
    def __init__(self):

        self.u = UtilsARC()
    
    
    def intervene_color(self,
                        input_grid: np.array, 
                        color_map: dict,
                        plot: bool = True,
                        figsize: tuple = (10,4),
                        subplot_titles: list = ["Factual", "Counterfactual"]) -> np.array:

        '''
        Intervene on pixel colors, given a color map.
        '''
    
        # Update input grid colors.
        cf_grid = input_grid.copy()
        for key,val in color_map.items():
            if key in cf_grid:
                cf_grid[cf_grid == key] = val
    
        # Plot counterfactual alongside original.
        if plot:
            self.u.plot_input_output(input_grid, 
                                     cf_grid, 
                                     figsize = figsize, 
                                     subplot_titles = subplot_titles)
        
        return cf_grid


    def intervene_geometric(self,
                            input_grid: np.array, 
                            transformation: str = "rot90", # "fliplr", "flipud"
                            plot: bool = True,
                            figsize: tuple = (10,4),
                            subplot_titles: list = ["Factual", "Counterfactual"]) -> np.array:

        '''
        Perform geometric interventions: 90 degree rotation, flipping
        left-to-right, or flipping up-to-down.
        '''
    
        # Perform geometric transformation.
        cf_grid = input_grid.copy()
        if transformation == "rot90":
            cf_grid = np.rot90(cf_grid)
        elif transformation == "fliplr":
            cf_grid = np.fliplr(cf_grid)
        elif transformation == "flipud":
            cf_grid = np.flipud(cf_grid)
        else:
            raise ValueError("Arg `transformation` must be in ['rot90', 'fliplr', 'flipud']")
    
        # Plot counterfactual alongside original.
        if plot:
            self.u.plot_input_output(input_grid, 
                                     cf_grid, 
                                     figsize = figsize, 
                                     subplot_titles = subplot_titles)
        
        return cf_grid


    def intervene_composition(self,
                              input_grids: list, 
                              axis: int = 1,
                              n_replicates: int = 2,
                              pad: bool = False,
                              pad_color: int = 10, #white
                              plot: bool = True,
                              subplot_titles: list = ["Grid composition", "Input grids"]) -> np.array:

        '''
        Concatenate multiple input grids, or replicate a single input grid.

        If inputting a single grid, this must still be as a single-item list.
        '''

        cf_grids = input_grids.copy()
        
        # Add borders separating subarrays, if desired.
        if pad:
            cf_grids = [np.pad(x, pad_width=1, constant_values=pad_color) for x in cf_grids]
        
        # Concatenate along desired axis.
        if len(cf_grids) > 1:
            cf_grid = np.concatenate(cf_grids, axis = axis)
            if plot:
                self.u.plot_single(cf_grid, 
                                   title = subplot_titles[0], 
                                   figsize = (3*len(input_grids), 2))
                self.u.plot_multi(input_grids, 
                                  title = subplot_titles[1])
        else:
            cf_grid = np.concatenate(cf_grids*n_replicates, axis = axis)
            if plot:
                self.u.plot_single(cf_grid, 
                                   title = subplot_titles[0], 
                                   figsize = (3*n_replicates, 2))
                self.u.plot_single(input_grids[0], 
                                   title = subplot_titles[1], 
                                   figsize = (2, 2))

        
        
        return cf_grid


    def intervene_resolution(self,
                             input_grid: np.array, 
                             factor: int = 2,
                             plot: bool = True,
                             figsize: tuple = (10,4),
                             subplot_titles: list = ["Factual", "Counterfactual"]) -> np.array:

        '''
        Increase resolution by an integer factor (e.g., factor = 2 -> duplicate all pixels).
        '''

        
        cf_grid = np.repeat(input_grid, factor, axis = 0)
        cf_grid = np.repeat(cf_grid, factor, axis = 1)
        
        # Plot counterfactual alongside original.
        if plot:
            self.u.plot_input_output(input_grid, 
                                     cf_grid, 
                                     figsize = figsize, 
                                     subplot_titles = subplot_titles)

        return cf_grid
        

    def get_sprite(self,
                   background_color: int = 0,
                   sprite_color: int = 1,
                   sprite_size: tuple = (3,3),
                   plot: bool = False) -> np.array:

        '''
        Construct a random sprite of a given size and color.
        '''
    
        sprite = np.random.choice([background_color, sprite_color], 
                                  sprite_size, 
                                  p = [0.5, 0.5])
        if plot:
            u.plot_single(sprite)
            
        return sprite


    def add_sprite(self,
                   input_grid: np.array,
                   output_grid: np.array = None, # If you want the output grid to get the sprite in the exact same place.
                   sprite: np.array = None, # If not random sprite.
                   output_sprite: np.array = None, # If not random sprite.
                   background_color: int = 0,
                   sprite_color: int = 1,
                   sprite_size: tuple = (3,3),
                   sprite_location: tuple = None,
                   plot: bool = False) -> np.array:

        '''
        Blit sprite onto background grid.
        '''

        cf_input = input_grid.copy()
        if sprite is None:
            sprite = self.get_sprite(background_color = background_color,
                                     sprite_color = sprite_color,
                                     sprite_size = sprite_size)
        if output_grid is not None and output_sprite is None:
            output_sprite = sprite.copy()

        if sprite_location is None:
            x, y = self.random_free_location_for_sprite(input_grid, 
                                                        sprite, 
                                                        padding = 1, 
                                                        padding_connectivity = 8, 
                                                        border_size = 1, 
                                                        background = background_color) 
            assert not self.collision(object1 = input_grid, object2 = sprite, x2 = x, y2 = y)
        else:
            x = sprite_location[0]
            y = sprite_location[1]

        # From blit() in 
        # https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py#L242
        for i in range(sprite.shape[0]):
                for j in range(sprite.shape[1]):
                    if background_color is None or sprite[i, j] != background_color:
                        # check that it is inbounds
                        if 0 <= x + i < input_grid.shape[0] and 0 <= y + j < input_grid.shape[1]:
                            cf_input[x + i, y + j] = sprite[i, j]

        if output_grid is not None:
            cf_output = output_grid.copy()
            for i in range(output_sprite.shape[0]):
                for j in range(output_sprite.shape[1]):
                    if background_color is None or output_sprite[i, j] != background_color:
                        # check that it is inbounds
                        if 0 <= x + i < output_grid.shape[0] and 0 <= y + j < output_grid.shape[1]:
                            cf_output[x + i, y + j] = output_sprite[i, j]
                            
        if plot:
            print("\nInput Grid:")
            self.u.plot_input_output(input_grid, 
                                     cf_input,
                                     subplot_titles = ["Original", "+ Sprite"])
            if output_grid is not None:
                print("\nOutput Grid:")
                self.u.plot_input_output(ouput_grid, 
                                         cf_output,
                                         subplot_titles = ["Original", "+ Sprite"])
        
        if output_grid is None:
            return cf_input
        else:
            return cf_input,cf_output


    def random_free_location_for_sprite(self,
                                        grid: np.array,
                                        sprite: np.array,
                                        background: int = 0,
                                        border_size: int = 0,
                                        padding: int = 0,
                                        padding_connectivity: int = 8):

        '''
        # From https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py
        Find a random free location for the sprite in the grid
        Returns a tuple (x, y) of the top-left corner of the sprite in the grid, 
        which can be passed to `blit_sprite`
    
        border_size: minimum distance from the edge of the grid
        background: color treated as transparent
        padding: if non-zero, the sprite will be padded with a non-background 
        color before checking for collision
        padding_connectivity: 4 or 8, for 4-way or 8-way connectivity when padding the sprite
    
        Example usage:
        # find the location, using generous padding
        x, y = random_free_location_for_sprite(grid, sprite, 
                            padding=1, padding_connectivity=8, 
                            border_size=1, background=Color.BLACK) 
        assert not collision(object1=grid, object2=sprite, x2=x, y2=y)
        blit_sprite(grid, sprite, x, y)
    
        If no free location can be found, raises a ValueError.
        '''
        
        n, m = grid.shape
    
        sprite_mask = 1 * (sprite != background)
    
        # If padding is non-zero, we emulate padding by dilating 
        # everything within the grid.
        if padding > 0:
            from scipy import ndimage
    
            if padding_connectivity == 4:
                structuring_element = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif padding_connectivity == 8:
                structuring_element = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
            else:
                raise ValueError("padding_connectivity must be 4 or 8.")
    
            # use binary dilation to pad the sprite with a non-background color
            grid_mask = ndimage.binary_dilation(
                grid != background, iterations=padding, structure=structuring_element
            ).astype(int)
        else:
            grid_mask = 1 * (grid != background)
    
        possible_locations = [
            (x, y)
            for x in range(border_size, n + 1 - border_size - sprite.shape[0])
            for y in range(border_size, m + 1 - border_size - sprite.shape[1])
        ]
    
        non_background_grid = np.sum(grid_mask)
        non_background_sprite = np.sum(sprite_mask)
        target_non_background = non_background_grid + non_background_sprite
    
        # Scale background pixels to 0 so np.maximum can be used later.
        scaled_grid = grid.copy()
        scaled_grid[scaled_grid == background] = 0 #Color.BLACK
    
        # Prune possible locations by making sure there is no overlap 
        # with non-background pixels if we were to put the sprite there.
        pruned_locations = []
        for x, y in possible_locations:
            # Try blitting the sprite and see if the resulting 
            # non-background pixels is the expected value.
            new_grid_mask = grid_mask.copy()
            self.blit(new_grid_mask, sprite_mask, x, y, background=0)
            if np.sum(new_grid_mask) == target_non_background:
                pruned_locations.append((x, y))

        # Return indices.
        if len(pruned_locations) == 0:
            raise ValueError("No free location for sprite found.")
        idx = np.random.choice(len(pruned_locations),1)[0]
        return pruned_locations[idx]


    def blit(self,
             grid: np.array, 
             sprite: np.array,
             x: int = 0, 
             y: int = 0, 
             background = None):

        '''
        # From https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py
        Copies the sprite into the grid at the specified location. Modifies the grid in place.
    
        background: color treated as transparent. 
            If specified, only copies the non-background pixels of the sprite.
        '''
    
        new_grid = grid
    
        x, y = int(x), int(y)
    
        for i in range(sprite.shape[0]):
            for j in range(sprite.shape[1]):
                if background is None or sprite[i, j] != background:
                    # check that it is inbounds
                    if 0 <= x + i < grid.shape[0] and 0 <= y + j < grid.shape[1]:
                        new_grid[x + i, y + j] = sprite[i, j]
    
        return new_grid
        

    def collision(self,
                  #_ = None, # unused?
                  object1: np.array = None, 
                  object2: np.array = None, 
                  x1: int = 0, 
                  y1: int = 0, 
                  x2: int = 0, 
                  y2: int = 0, 
                  background: int = 0) -> bool:
        
        '''
        # From https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py
        
        Check if object1 and object2 collide when object1 is at (x1, y1) and object2 is at (x2, y2).
    
        Example usage:
    
        # Check if a sprite can be placed onto a grid at (X,Y)
        collision(object1=output_grid, object2=a_sprite, x2=X, y2=Y)
    
        # Check if two objects collide
        collision(object1=object1, object2=object2, x1=X1, y1=Y1, x2=X2, y2=Y2)
        '''
        
        n1, m1 = object1.shape
        n2, m2 = object2.shape
    
        dx = x2 - x1
        dy = y2 - y1
        dx, dy = int(dx), int(dy)
    
        for x in range(n1):
            for y in range(m1):
                if object1[x, y] != background:
                    new_x = x - dx
                    new_y = y - dy
                    if (
                        0 <= new_x < n2
                        and 0 <= new_y < m2
                        and object2[new_x, new_y] != background
                    ):
                        return True
    
        return False


    def crop(self,
             grid: np.array, 
             background: int = 0) -> np.array:

        '''
        # From https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py
        Crop the grid to the smallest bounding box that contains all non-background pixels.
    
        Example usage:
        # Extract a sprite from an object
        sprite = crop(an_object, background=background_color)
        '''
        
        x, y, w, h = self.bounding_box(grid, background)
        return grid[x : x + w, y : y + h]

    
    def bounding_box(self,
                     grid: np.array, 
                     background: int = 0,
                     foreground: int = None):

        '''
        # From https://github.com/YRIKKA/ttt_barc/blob/main/seeds/common.py
        Find the bounding box of the non-background pixels in the grid.
        Returns a tuple (x, y, width, height) of the bounding box.
    
        Example usage:
        objects = find_connected_components(input_grid, monochromatic=True, 
                    background=Color.BLACK, connectivity=8)
        teal_object = [ obj for obj in objects if np.any(obj == Color.TEAL) ][0]
        teal_x, teal_y, teal_w, teal_h = bounding_box(teal_object)
        '''
        
        n, m = grid.shape
        x_min, x_max = n, -1
        y_min, y_max = m, -1
    
        for x in range(n):
            for y in range(m):
                if foreground is not None:
                    if grid[x, y] == foreground:
                        x_min = min(x_min, x)
                        x_max = max(x_max, x)
                        y_min = min(y_min, y)
                        y_max = max(y_max, y)
                else:
                    if grid[x, y] != background:
                        x_min = min(x_min, x)
                        x_max = max(x_max, x)
                        y_min = min(y_min, y)
                        y_max = max(y_max, y)
    
        return x_min, y_min, x_max - x_min + 1, y_max - y_min + 1

