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


class TaskOrder:

    
    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()

        # Dictionary mapping CausalARC tasks to the ARC task that inspired them.
        # Tasks may only be loosely based on each other.
        # Tasks not inspired by a specific ARC task map to None. 
        self.reference_dict = {"SCMv4bg": "575b1a71",  
                               "SCMye7g": "5af49b42", 
                               "SCMswcs": "66e6c45b", 
                               "SCMtzlq": None, 
                               "SCMffb8": None}
        

    def task_SCMv4bg(self,
                     size: int = 10,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC-AGI-1 task 575b1a71.
        '''

        def SCMv4bg(size: tuple = (10,9)):
            _u = np.random.binomial(n = 1, p = 0.6, size = size)
            x = 9*_u
            y = x.copy()
            for j in range(y.shape[1]):
                y[:,j][y[:,j] == 0] = j
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMv4bg(size: tuple = (10,9)):
            _u = np.random.binomial(n = 1, p = 0.6, size = size)
            x = 9*_u
            y = x.copy()
            for j in range(y.shape[1]):
                y[:,j][y[:,j] == 0] = j
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        size = (size,9)
        x_test,y_test = SCMv4bg(size)
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
            x,y = SCMv4bg(size)

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
            
            # Hard interventions: change some columns to 9.
            x_color = x.copy()
            jump = 2
            x_color[:,range(0,size[1],jump)] = 9
            y_color = y.copy()
            y_color[:,range(0,size[1],jump)] = 9
            cf_inputs.append(x_color)
            cf_outputs.append(y_color)
            cf_types.append(f"do_hard (every {jump} column = 9)")

            x_color = x.copy()
            jump = 3
            x_color[:,range(0,size[1],jump)] = 9
            y_color = y.copy()
            y_color[:,range(0,size[1],jump)] = 9
            cf_inputs.append(x_color)
            cf_outputs.append(y_color)
            cf_types.append(f"do_hard (every {jump} column = 9)")
    
            # Soft intervention on geometry.
            for transform in [np.flipud, np.fliplr, np.rot90]:
                x_transform = transform(x.copy())
                cf_inputs.append(x_transform)
                cf_outputs.append(y.copy())
                cf_types.append(f"do_soft (geometric transform = {transform.__name__})")

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


    def task_SCMye7g(self,
                     size: int = 15,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC-AGI-1 task 5af49b42.
        '''

        def SCMye7g(n: int = 10):
            size = (n-2,n)
            colors = list(range(1,10))
            order = np.random.choice(colors, 
                                     size = 3, #np.random.randint(low = 3, high = 7),
                                     replace = False)
            final_row = list(order)+[0]*(n-len(order))
            
            x = np.zeros(size).astype(int)
            for i in range(size[0]):
                j = np.random.randint(low = 0, high = size[1])
                x[i,j] = np.random.choice(order, size = 1)[0]
            x = np.concatenate((x,np.zeros((1,n)))).astype(int)
            x = np.concatenate((x,np.array(final_row).reshape(1,-1)), axis = 0)
            
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i,j] == 0:
                        continue
                    elif y[i,j] == order[0]:
                        try:
                            y[i,j+1] = order[1]
                            y[i,j+2] = order[2]
                        except:
                            pass
                    elif y[i,j] == order[1]:
                        try:
                            y[i,j+1] = order[2]
                            y[i,j-1] = order[0]
                        except:
                            pass
                    elif y[i,j] == order[2]:
                        try:
                            y[i,j-1] = order[1]
                            y[i,j-2] = order[0]
                        except:
                            pass
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMye7g(n: int = 10):
            size = (n-2,n)
            colors = list(range(1,10))
            order = np.random.choice(colors, 
                                     size = 3, #np.random.randint(low = 3, high = 7),
                                     replace = False)
            final_row = list(order)+[0]*(n-len(order))
            
            x = np.zeros(size).astype(int)
            for i in range(0,size[0],2):
                j = np.random.randint(low = 0, high = size[1])
                x[i,j] = np.random.choice(order, size = 1)[0]
            x = np.concatenate((x,np.zeros((1,n)))).astype(int)
            x = np.concatenate((x,np.array(final_row).reshape(1,-1)), axis = 0)
            
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if y[i,j] == 0:
                        continue
                    elif y[i,j] == order[0]:
                        try:
                            y[i,j+1] = order[1]
                            y[i,j+2] = order[2]
                        except:
                            pass
                    elif y[i,j] == order[1]:
                        try:
                            y[i,j+1] = order[2]
                            y[i,j-1] = order[0]
                        except:
                            pass
                    elif y[i,j] == order[2]:
                        try:
                            y[i,j-1] = order[1]
                            y[i,j-2] = order[0]
                        except:
                            pass
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMye7g(n = size)
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
            x,y = SCMye7g(n = size)

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
    
            # Soft interventions on colors.
            colors = np.unique(x)
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
            for transform in [np.flipud, np.fliplr, np.rot90]:
                x_transform = transform(x.copy())
                y_transform = transform(y.copy())
                cf_inputs.append(x_transform)
                cf_outputs.append(y_transform)
                cf_types.append(f"do_soft (geometric transform = {transform.__name__})")

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
                        

    def task_SCMswcs(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC-AGI-1 task 66e6c45b.
        '''

        def SCMswcs():
            input_grids = []
            output_grids = []
            n = np.random.randint(low = 4, high = 8, size = 1)[0]
            for i in range(n):
                _u = np.random.choice(list(range(1,10)), size = (2,2), replace = False)
                x = np.pad(_u, pad_width = 1)
                y = x.copy()
                y[0,0] = x[1,1]
                y[3,0] = x[2,1]
                y[0,3] = x[1,2]
                y[3,3] = x[2,2]
                input_grids.append(x)
                output_grids.append(y)
            x = np.concatenate(input_grids, axis = 1)
            y = np.concatenate(output_grids, axis = 1)
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMswcs(n: int = 4):
            input_grids = []
            output_grids = []
            for i in range(n):
                _u = np.random.choice(list(range(1,10)), size = (2,2), replace = False)
                x = np.pad(_u, pad_width = 1)
                y = x.copy()
                y[0,0] = x[1,1]
                y[3,0] = x[2,1]
                y[0,3] = x[1,2]
                y[3,3] = x[2,2]
                input_grids.append(x)
                output_grids.append(y)
            x = np.concatenate(input_grids, axis = 1)
            y = np.concatenate(output_grids, axis = 1)
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMswcs()
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
            x,y = SCMswcs()

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
    
            # Soft interventions on colors.
            colors = np.unique(x)
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
            for transform in [np.flipud, np.fliplr]:
                x_transform = transform(x.copy())
                y_transform = transform(y.copy())
                cf_inputs.append(x_transform)
                cf_outputs.append(y_transform)
                cf_types.append(f"do_soft (geometric transform = {transform.__name__})")

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

    
    def task_SCMtzlq(self,
                     size: tuple = (20,20),
                     n_sprites: int = 8,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Ordering sprites.
        '''
        
        if n_sprites > 9:
            raise ValueError("n_sprites must be less than 10.")
        
        def SCMtzlq(n: int = 8, size: tuple = (20,20)):
            sprites = []
            order_row = np.zeros((2,size[1])).astype(int)
            order = np.random.choice(range(1,10), size = n, replace = False)
            order_row[1,:n] = order
            
            x = np.zeros((size[0]-2,size[1])).astype(int)
            for i in range(n):
                sprite = np.random.binomial(n = 1, size = (3,3), p = 0.8)*order[i]
                sprites.append(sprite)
                x = self.a.add_sprite(input_grid = x, sprite = sprite)
            x = np.concatenate((x,order_row))
            y = np.concatenate(sprites, axis = 1)
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMtzlq(n: int = 8, size: tuple = (20,20)):
            sprites = []
            order_row = np.zeros((2,size[1])).astype(int)
            order = np.random.choice(range(1,10), size = n, replace = False)
            order_row[1,:n] = order
            
            x = np.zeros((size[0]-2,size[1])).astype(int)
            for i in range(n):
                sprite = np.random.binomial(n = 1, size = (3,3), p = 0.8)*order[i]
                sprites.append(sprite)
                x = self.a.add_sprite(input_grid = x, sprite = sprite)
            x = np.concatenate((x,order_row))
            y = np.concatenate(sprites, axis = 1)
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMtzlq(n = n_sprites, size = size)
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
            x,y = SCMtzlq(n = n_sprites, size = size)

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
    
            # Soft interventions on colors.
            colors = np.unique(x)
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
            for transform in [np.flipud, np.fliplr]:
                x_transform = transform(x.copy())
                y_transform = transform(y.copy())
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


    def task_SCMffb8(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        def SCMffb8():
            size = (15,15)
            sprite_sizes = [1,2,3,4]
            order = np.random.choice([1,2,3,4,6,7,8,9], size = 4, replace = False)
            order_rows = np.zeros((4,size[1])).astype(int)
            order_rows[-1,0] = order[0]
            order_rows[-2:,1] = order[1]
            order_rows[-3:,2] = order[2]
            order_rows[-4:,3] = order[3]
            x = np.zeros((size[0]-4,size[1])).astype(int)
            sprites = []
            for i in range(len(sprite_sizes)):
                if sprite_sizes[i] == 1:
                    sprite = np.array([[order[i]]])
                elif sprite_sizes[i] == 2:
                    sprite = np.ones((2,2)).astype(int)*order[i]
                else:
                    sprite = np.random.binomial(n = 1, size = (sprite_sizes[i],sprite_sizes[i]), p = 0.8)*order[i]
                sprites.append(sprite)
                x = self.a.add_sprite(input_grid = x, sprite = sprite)
            y = x.copy()
            for color in order:
                x[x==color] = 5
            x = np.concatenate((x,order_rows))
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMffb8():
            sprite_sizes = [1,2,3,4]
            order = np.random.choice([1,2,3,4,6,7,8,9], size = 4, replace = False)
            order_rows = np.zeros((4,size[1])).astype(int)
            order_rows[-1,0] = order[0]
            order_rows[-2:,1] = order[1]
            order_rows[-3:,2] = order[2]
            order_rows[-4:,3] = order[3]
            x = np.zeros((size[0]-4,size[1])).astype(int)
            sprites = []
            for i in range(len(sprite_sizes)):
                if sprite_sizes[i] == 1:
                    sprite = np.array([[order[i]]])
                elif sprite_sizes[i] == 2:
                    sprite = np.ones((2,2)).astype(int)*order[i]
                else:
                    sprite = np.random.binomial(n = 1, size = (sprite_sizes[i],sprite_sizes[i]), p = 0.8)*order[i]
                sprites.append(sprite)
                x = self.a.add_sprite(input_grid = x, sprite = sprite)
            y = x.copy()
            for color in order:
                x[x==color] = 5
            x = np.concatenate((x,order_rows))
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMffb8()
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
            x,y = SCMffb8()

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
    
            # Soft interventions on colors.
            colors = [x for x in np.unique(x) if x not in [0,5]]
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
            for transform in [np.flipud, np.fliplr, np.rot90]:
                x_transform = transform(x.copy())
                y_transform = transform(y.copy())
                cf_inputs.append(x_transform)
                cf_outputs.append(y_transform)
                cf_types.append(f"do_soft (geometric transform = {transform.__name__})")

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

            
