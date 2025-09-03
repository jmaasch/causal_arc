# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import math
import random

# Custom modules.
from carc_utils import UtilsARC
from carc_augment import AugmentARC


class TaskSprites:

    
    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()

        # Dictionary mapping CausalARC tasks to the ARC task that inspired them.
        # Tasks may only be loosely based on each other.
        # Tasks not inspired by a specific ARC task map to None. 
        self.reference_dict = {"SCMrwf3": "e74e1818", 
                               "SCMw3c1": "0bb8deee", 
                               "SCMcxdz": "e7dd8335", 
                               "SCMvyii": None, 
                               "SCM1smt": None}

        
    def task_SCMrwf3(self,
                     n_examples: int = 5,
                     n_sprites: int = 3, # How many sprites to stack.
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True
                     ) -> dict:

        '''
        Based on ARC task e74e1818.
        '''

        # Get str representation of SCM.
        scm_str = "def get_xy(colors, sprite_sizes):\n"
        scm_str += "\tsprites = []\n"
        scm_str += "\tfor color,sprite_size in zip(colors,sprite_sizes):\n"
        scm_str += "\t\tz = np.random.choice([0, color], size = sprite_size, p = (0.4,0.6))\n"
        scm_str += "\t\tz = np.concatenate((z, np.fliplr(z)), axis = 1)\n"
        scm_str += "\t\tsprites.append(z)\n"
        scm_str += "\tx = np.concatenate(sprites, axis = 0)\n"
        scm_str += "\tx = np.pad(x, pad_width = 1)\n"
        scm_str += "\tsprites_ud = [np.flipud(grid) for grid in sprites]\n"
        scm_str += "\ty = np.concatenate(sprites_ud, axis = 0)\n"
        scm_str += "\ty = np.pad(y, pad_width = 1)\n"
        scm_str += "\treturn x,y"
        sample_dict = {"scm": scm_str}

        # Helper functions.
        def get_xy(colors, sprite_sizes):
            sprites = []
            for color,sprite_size in zip(colors,sprite_sizes):
                z = np.random.choice([0, color], size = sprite_size, p = (0.4,0.6))
                z = np.concatenate((z, np.fliplr(z)), axis = 1)
                sprites.append(z)
            x = np.concatenate(sprites, axis = 0)
            x = np.pad(x, pad_width = 1)
            sprites_ud = [np.flipud(grid) for grid in sprites]
            y = np.concatenate(sprites_ud, axis = 0)
            y = np.pad(y, pad_width = 1)
            return x,y

        # Get test sample.
        colors = np.random.choice(list(range(1,10)), size = n_sprites, replace = False)
        random.shuffle(colors)
        width = np.random.randint(low = 4, high = 8, size = 1)[0]
        sprite_sizes = [(np.random.randint(low = 2, high = 5),width) for _ in range(len(colors))]
        x_test,y_test = get_xy(colors, sprite_sizes)
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        if plot:
            print("Test pair")
            if plot_type == "single":
                self.u.plot_single(input_grid = x_test, grid = grid, figsize = figsize)
                self.u.plot_single(input_grid = y_test, grid = grid, figsize = figsize)
            else:
                self.u.plot_input_output(input_grid = x_test, 
                                         output_grid = y_test, 
                                         grid = grid, 
                                         figsize = figsize)
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            colors = np.random.choice(list(range(1,10)), size = n_sprites, replace = False)
            random.shuffle(colors)
            width = np.random.randint(low = 4, high = 8, size = 1)[0]
            sprite_sizes = [(np.random.randint(low = 2, high = 5),width) for _ in range(len(colors))]
            x,y = get_xy(colors, sprite_sizes)

            if plot:
                print("Demonstration pair")
                if plot_type == "single":
                    self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                else:
                    self.u.plot_input_output(input_grid = x, 
                                             output_grid = y, 
                                             grid = grid, 
                                             figsize = figsize)
            cf_inputs = []
            cf_outputs = []
            cf_types = []
            
            # Hard interventions: color -> 0.
            for color in colors:
                x_color = x.copy()
                x_color[x_color == color] = 0
                y_color = y.copy()
                y_color[y_color == color] = 0
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_hard (color {color} = 0)")
    
            # Soft interventions on colors.
            cf_colors = [x for x in range(1,10) if x not in colors]
            random.shuffle(cf_colors)
            cf_colors = cf_colors[:4]
            for color,cf_color in zip(colors,cf_colors):
                x_color = x.copy()
                x_color[x_color == color] = cf_color
                y_color = y.copy()
                y_color[y_color == color] = cf_color
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_soft (color {color} = {cf_color})")
    
            # Soft intervention on geometry.
            for transform in [np.flipud]:
                x_transform = x.copy()
                x_transform = transform(x_transform)
                y_transform = y.copy()
                y_transform = transform(y_transform)
                cf_inputs.append(x_transform)
                cf_outputs.append(y_transform)
                cf_types.append(f"do_soft (geometric transform = {transform})")

            if plot:
                for input_grid,output_grid,name in zip(cf_inputs,cf_outputs,cf_types):
                    if plot_type == "single":
                        print(name)
                        self.u.plot_single(input_grid = input_grid, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = output_grid, grid = grid, figsize = figsize)
                    else:
                        print(name)
                        self.u.plot_input_output(input_grid = input_grid, 
                                                 output_grid = output_grid, 
                                                 grid = grid, 
                                                 figsize = figsize)
    
            train_list.append(
                    {"input": x.tolist(),
                     "output": y.tolist(), 
                     "cf_types": cf_types, 
                     "cf_inputs": [x.tolist() for x in cf_inputs],
                     "cf_outputs": [x.tolist() for x in cf_outputs]}
                                 )
            
        sample_dict["train"] = train_list
        return sample_dict


    def task_SCMw3c1(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True
                     ) -> dict:


        '''
        Based on ARC task 0bb8deee.
        '''
        
        # Get str representation of SCM.
        scm_str = "def get_xy():\n"
        scm_str += "\tcolors = np.random.choice(list(range(1,10)), size = n_sprites)"
        scm_str += "\trandom.shuffle(colors)"
        scm_str += "\t_u = [np.random.binomial(n = 1, p = 0.75, size = (3,3)) for _ in range(4)]\n"
        scm_str += "\tc = np.random.choice(list(range(1,10)), size = len(_u), replace = False)\n"
        scm_str += "\tc_u = [c[i]*_u[i] for i in range(len(c))]\n"
        scm_str += "\tx_upper = [np.pad(c_u[i], pad_width = 2) for i in range(2)]\n"
        scm_str += "\tx_upper = np.concatenate(x_upper, axis = 1)\n"
        scm_str += "\tx_lower = [np.pad(c_u[i], pad_width = 2) for i in range(2,4)]\n"
        scm_str += "\tx_lower = np.concatenate(x_lower, axis = 1)\n"
        scm_str += "\tx = np.concatenate([x_upper,x_lower], axis = 0)\n"
        scm_str += "\ty_upper = np.concatenate([c_u[0],c_u[1]], axis = 1)\n"
        scm_str += "\ty_lower = np.concatenate([c_u[2],c_u[3]], axis = 1)\n"
        scm_str += "\ty = np.concatenate([y_upper,y_lower], axis = 0)\n"
        scm_str += "\treturn x,y"
        sample_dict = {"scm": scm_str}

        # Helper functions.
        def get_xy(sprite_size):
            colors = np.random.choice(list(range(1,10)), size = 4)
            random.shuffle(colors)
            _u = [np.random.binomial(n = 1, p = 0.75, size = (3,3)) for _ in range(4)]
            c = np.random.choice(list(range(1,10)), size = len(_u), replace = False)
            c_u = [c[i]*_u[i] for i in range(len(c))]
            
            x_upper = [np.pad(c_u[i], pad_width = 2) for i in range(2)]
            x_upper = np.concatenate(x_upper, axis = 1)
            x_lower = [np.pad(c_u[i], pad_width = 2) for i in range(2,4)]
            x_lower = np.concatenate(x_lower, axis = 1)
            x = np.concatenate([x_upper,x_lower], axis = 0)
            
            y_upper = np.concatenate([c_u[0],c_u[1]], axis = 1)
            y_lower = np.concatenate([c_u[2],c_u[3]], axis = 1)
            y = np.concatenate([y_upper,y_lower], axis = 0)
            return x,y

        # Get test sample.
        sprite_size = np.random.randint(low = 4, high = 8, size = 2)
        x_test,y_test = get_xy(sprite_size)
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        if plot:
            print("Test pair")
            if plot_type == "single":
                self.u.plot_single(input_grid = x_test, grid = grid, figsize = figsize)
                self.u.plot_single(input_grid = y_test, grid = grid, figsize = figsize)
            else:
                self.u.plot_input_output(input_grid = x_test, 
                                         output_grid = y_test, 
                                         grid = grid, 
                                         figsize = figsize)
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            sprite_size = np.random.randint(low = 4, high = 8, size = 2)
            x,y = get_xy(sprite_size)

            if plot:
                print("Demonstration pair")
                if plot_type == "single":
                    self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                else:
                    self.u.plot_input_output(input_grid = x, 
                                             output_grid = y, 
                                             grid = grid, 
                                             figsize = figsize)
            cf_inputs = []
            cf_outputs = []
            cf_types = []
            
            # Hard interventions: color -> 0.
            colors = [c for c in np.unique(x) if c > 0]
            for color in colors:
                x_color = x.copy()
                x_color[x_color == color] = 0
                y_color = y.copy()
                y_color[y_color == color] = 0
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_hard (color {color} = 0)")
    
            # Soft interventions on colors.
            cf_colors = [c for c in range(1,10) if c not in colors]
            random.shuffle(cf_colors)
            cf_colors = cf_colors[:4]
            for color,cf_color in zip(colors,cf_colors):
                x_color = x.copy()
                x_color[x_color == color] = cf_color
                y_color = y.copy()
                y_color[y_color == color] = cf_color
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_soft (color {color} = {cf_color})")
    
            # Soft intervention on geometry.
            for transform in [np.flipud, np.fliplr, np.rot90]:
                x_transform = x.copy()
                x_transform = transform(x_transform)
                y_transform = y.copy()
                y_transform = transform(y_transform)
                cf_inputs.append(x_transform)
                cf_outputs.append(y_transform)
                cf_types.append(f"do_soft (geometric transform = {transform})")

            if plot:
                for input_grid,output_grid,name in zip(cf_inputs,cf_outputs,cf_types):
                    if plot_type == "single":
                        print(name)
                        self.u.plot_single(input_grid = input_grid, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = output_grid, grid = grid, figsize = figsize)
                    else:
                        print(name)
                        self.u.plot_input_output(input_grid = input_grid, 
                                                 output_grid = output_grid, 
                                                 grid = grid, 
                                                 figsize = figsize)
                        
            train_list.append(
                    {"input": x.tolist(),
                     "output": y.tolist(), 
                     "cf_types": cf_types, 
                     "cf_inputs": [x.tolist() for x in cf_inputs],
                     "cf_outputs": [x.tolist() for x in cf_outputs]}
                                 )
            
        sample_dict["train"] = train_list
        return sample_dict


    
    def task_SCMcxdz(colors: tuple = (6,7), 
                     size: tuple = (6,6)):

        '''
        Based on ARC task e7dd8335.
        '''
    
        def get_xy(colors: tuple = (6,7), size: tuple = (6,6)):
            if (size[0] % 2 != 0) or (size[1] % 2 != 0):
                raise ValueError("Both elements in size must be even.")
            
            _u = np.random.binomial(n = 1, p = 0.75, size = size)
            x = _u*colors[0]
            y = x.copy()
            half = size[0]//2
            y[half:,:][(y[half:,:]==colors[0])] = colors[1]
            x = np.pad(x, pad_width = 2)
            y = np.pad(y, pad_width = 2)
            return x,y

        scm = '''def get_xy(colors: tuple = (6,7), size: tuple = (6,6)):
            if (size[0] % 2 != 0) or (size[1] % 2 != 0):
                raise ValueError("Both elements in size must be even.")
            
            _u = np.random.binomial(n = 1, p = 0.75, size = size)
            x = _u*colors[0]
            y = x.copy()
            half = size[0]//2
            y[half:,:][(y[half:,:]==colors[0])] = colors[1]
            x = np.pad(x, pad_width = 2)
            y = np.pad(y, pad_width = 2)
            return x,y
        '''
        sample_dict = {"scm": scm}
        

    def task_SCMvyii(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True
                     ) -> dict:


        def get_xy(colors: tuple = (6,8), size: tuple = (6,6)):
            if (size[0] % 2 != 0) or (size[1] % 2 != 0):
                raise ValueError("Both elements in size must be even.")
                
            _u = np.random.binomial(n = 1, p = 0.6, size = size)
            x = _u.copy()
            #x[x==0] = colors[0]
            y = x.copy()
            half = size[0]//2
            y[half:,:][(y[half:,:]==0)] = colors[0]
            y[y==0] = colors[1]
            x = np.pad(x, pad_width = 1)
            y = np.pad(y, pad_width = 1)
            return x,y

        scm = '''def get_xy(colors: tuple = (6,8), size: tuple = (6,6)):
            if (size[0] % 2 != 0) or (size[1] % 2 != 0):
                raise ValueError("Both elements in size must be even.")
                
            _u = np.random.binomial(n = 1, p = 0.6, size = size)
            x = _u.copy()
            #x[x==0] = colors[0]
            y = x.copy()
            half = size[0]//2
            y[half:,:][(y[half:,:]==0)] = colors[0]
            y[y==0] = colors[1]
            x = np.pad(x, pad_width = 1)
            y = np.pad(y, pad_width = 1)
            return x,y
        '''
        sample_dict = {"scm": scm}

    
    def task_SCM1smt(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True
                     ) -> dict:

        def get_xy(colors: tuple = (6,7,4,3,1,2), 
                   size: tuple = (3,3)):
    
            u_list = [np.random.binomial(n = 1, p = 0.6, size = size) for i in range(len(colors))]
            x = [u_list[i]*colors[i] for i in range(len(colors))]
            y = x.copy()
            x = [np.pad(z, pad_width = 1) for z in x]
            x = np.concatenate(x, axis = 1)
            y = np.concatenate(y, axis = 0)
            return x,y

        scm = '''def get_xy(colors: tuple = (6,8), size: tuple = (6,6)):
            if (size[0] % 2 != 0) or (size[1] % 2 != 0):
                raise ValueError("Both elements in size must be even.")
                
            _u = np.random.binomial(n = 1, p = 0.6, size = size)
            x = _u.copy()
            #x[x==0] = colors[0]
            y = x.copy()
            half = size[0]//2
            y[half:,:][(y[half:,:]==0)] = colors[0]
            y[y==0] = colors[1]
            x = np.pad(x, pad_width = 1)
            y = np.pad(y, pad_width = 1)
            return x,y
        '''
        sample_dict = {"scm": scm}

        # Vary size, color, and number of sprites.

