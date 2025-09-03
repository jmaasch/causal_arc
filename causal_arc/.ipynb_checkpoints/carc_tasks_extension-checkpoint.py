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


class TaskExtension:
    

    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()

        # Dictionary mapping CausalARC tasks to the ARC task that inspired them.
        # Tasks may only be loosely based on each other.
        # Tasks not inspired by a specific ARC task map to None. 
        self.reference_dict = {"SCMfwpq": "bcb3040b", 
                               "SCMig1o": "bcb3040b", 
                               "SCMz750": "705a3229"}

    
    def task_SCMfwpq(self,
                     size: tuple = (8,8), # (12,12), (16,16)
                     colors: list = [6,7,4],
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC task bcb3040b.
        '''


        scm = '''def SCMfwp(colors: list = [6,7,4], size: tuple = (8,8)):
            _u = np.random.binomial(n = 1, p = 0.25, size = size)
            x = colors[0]*_u
            for i in range(x.shape[0]):
                if i % modulo == 0:
                    x[i,0] = colors[1]
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if (i % modulo == 0) and (x[i,j] == 0):
                        y[i,j] = colors[1]
                    elif (i % modulo == 0) and (x[i,j] == colors[0]):
                        y[i,j] = colors[2]
                    else:
                        y[i,j] = x[i,j]
            return x,y
        '''
        sample_dict = {"scm": scm}

        def SCMfwp(colors: list = [6,7,4], size: tuple = (8,8), modulo: int = 5):
            _u = np.random.binomial(n = 1, p = 0.25, size = size)
            x = colors[0]*_u
            for i in range(x.shape[0]):
                if i % modulo == 0:
                    x[i,0] = colors[1]
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if (i % modulo == 0) and (x[i,j] == 0):
                        y[i,j] = colors[1]
                    elif (i % modulo == 0) and (x[i,j] == colors[0]):
                        y[i,j] = colors[2]
                    else:
                        y[i,j] = x[i,j]
            return x,y

        # Get test sample.
        modulo = np.random.choice([3,4,5], size = 1)[0]
        x_test,y_test = SCMfwp(colors = colors, size = size, modulo = modulo)
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            modulo = np.random.choice([3,4,5], size = 1)[0]
            x,y = SCMfwp(colors = colors, size = size, modulo = modulo)

            if plot:
                if plot_type == "single":
                    self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                else:
                    self.u.plot_input_output(input_grid = x, output_grid = y, grid = grid, figsize = figsize)

            #########################
            # Hard interventions.
            #########################

            cf_inputs = []
            cf_outputs = []
            cf_types = []
            
            # Extend row = 0.
            x_hard_0 = x.copy()
            y_hard_0 = y.copy()
            for i in range(x_hard_0.shape[0]):
                if i % modulo == 0:
                    x_hard_0[i,1:] = 0
            for i in range(y_hard_0.shape[0]):
                if i % modulo == 0:
                    y_hard_0[i,1:] = colors[1]
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
            cf_types.append("do_hard (extend row = 0)")

            # Extend row = colors[0].
            x_hard_1 = x.copy()
            y_hard_1 = y.copy()
            for i in range(x_hard_1.shape[0]):
                if i % modulo == 0:
                    x_hard_1[i,1:] = colors[0]
            for i in range(y_hard_1.shape[0]):
                if i % modulo == 0:
                    y_hard_1[i,1:] = colors[2]
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)
            cf_types.append("do_hard (extend row = colors[0])")

            # Unimportant row = 0.
            x_hard_0 = x.copy()
            y_hard_0 = y.copy()
            not_div_4 = [j for j in range(x.shape[0]) if (j % modulo != 0) and (len(np.unique(x[j,:])) > 1)]
            random_row = np.random.choice(not_div_4, size = 1)[0]
            x_hard_0[random_row,:] = 0
            y_hard_0[random_row,:] = 0
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
            cf_types.append("do_hard (random row = 0)")

            # Unimportant row = colors[0].
            x_hard_1 = x.copy()
            y_hard_1 = y.copy()
            not_div_4 = [j for j in range(x.shape[0]) if (j % modulo != 0) and (len(np.unique(x[j,:])) > 1)]
            random_row = np.random.choice(not_div_4, size = 1)[0]
            x_hard_1[random_row,:] = colors[0]
            y_hard_1[random_row,:] = colors[0]
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)
            cf_types.append("do_hard (random row = colors[0])")

            #########################
            # Soft interventions.
            #########################

            # Get random colors not used in original grids.
            cf_colors = [x for x in range(1,10) if x not in colors]
            random.shuffle(cf_colors)
            cf_colors = cf_colors[:3]
            for color in cf_colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==colors[0]] = color
                y_soft_color = y.copy()
                y_soft_color[y_soft_color==colors[0]] = color
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

            if plot:
                for x,y,cf_type in zip(cf_inputs,cf_outputs,cf_types):
                    if plot_type == "single":
                        print(cf_type)
                        self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                    else:
                        print(cf_type)
                        self.u.plot_input_output(input_grid = x, 
                                                 output_grid = y, 
                                                 grid = grid, 
                                                 figsize = figsize)
                    

            #assert len(cf_types) == len(cf_inputs) and len(cf_inputs) == len(cf_outputs)

            train_list.append(
                {"input": x.tolist(),
                 "output": y.tolist(), 
                 "cf_types": cf_types, 
                 "cf_inputs": [x.tolist() for x in cf_inputs],
                 "cf_outputs": [x.tolist() for x in cf_outputs]}
                             )
            
        sample_dict["train"] = train_list
        return sample_dict


    def task_SCMig1o(self,
                     size: tuple = (10,10), # (15,15), (20,20)
                     colors: list = [6,7,4],
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC task bcb3040b.
        '''

        scm = '''def SCMig1o(colors: list = [6,7,4], size: tuple = (10,10), modulo: int = 5):
            _u = np.random.binomial(n = 1, p = 0.25, size = size)
            x = colors[0]*_u
            for i in range(x.shape[1]):
                if i % modulo == 0:
                    x[0,i] = colors[1]
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if (j % modulo == 0) and (x[i,j] == 0):
                        y[i,j] = colors[1]
                    elif (j % modulo == 0) and (x[i,j] == colors[0]):
                        y[i,j] = colors[2]
                    else:
                        y[i,j] = x[i,j]
            return x,y
        '''
        sample_dict = {"scm": scm}

        def SCMig1o(colors: list = [6,7,4], size: tuple = (10,10), modulo: int = 5):
            _u = np.random.binomial(n = 1, p = 0.25, size = size)
            x = colors[0]*_u
            for i in range(x.shape[1]):
                if i % modulo == 0:
                    x[0,i] = colors[1]
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if (j % modulo == 0) and (x[i,j] == 0):
                        y[i,j] = colors[1]
                    elif (j % modulo == 0) and (x[i,j] == colors[0]):
                        y[i,j] = colors[2]
                    else:
                        y[i,j] = x[i,j]
            return x,y

        # Get test sample.
        modulo = np.random.choice([3,4,5], size = 1)[0]
        x_test,y_test = SCMig1o(colors = colors, size = size, modulo = modulo)
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            modulo = np.random.choice([3,4,5], size = 1)[0]
            x,y = SCMig1o(colors = colors, size = size, modulo = modulo)

            if plot:
                if plot_type == "single":
                    self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                else:
                    self.u.plot_input_output(input_grid = x, output_grid = y, grid = grid, figsize = figsize)

            #########################
            # Hard interventions.
            #########################

            cf_inputs = []
            cf_outputs = []
            cf_types = []
            
            # Extend col = 0.
            x_hard_0 = x.copy()
            y_hard_0 = y.copy()
            for i in range(x_hard_0.shape[1]):
                if i % modulo == 0:
                    x_hard_0[1:,i] = 0
            for i in range(y_hard_0.shape[1]):
                if i % modulo == 0:
                    y_hard_0[1:,i] = colors[1]
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
            cf_types.append("do_hard (extend col = 0)")

            # Extend col = colors[0].
            x_hard_1 = x.copy()
            y_hard_1 = y.copy()
            for i in range(x_hard_1.shape[1]):
                if i % modulo == 0:
                    x_hard_1[1:,i] = colors[0]
            for i in range(y_hard_1.shape[1]):
                if i % modulo == 0:
                    y_hard_1[1:,i] = colors[2]
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)
            cf_types.append("do_hard (extend col = colors[0])")

            # Unimportant col = 0.
            x_hard_0 = x.copy()
            y_hard_0 = y.copy()
            not_div_5 = [j for j in range(x.shape[1]) if (j % modulo != 0) and (len(np.unique(x[:,j])) > 1)]
            random_col = np.random.choice(not_div_5, size = 1)[0]
            x_hard_0[:,random_col] = 0
            y_hard_0[:,random_col] = 0
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
            cf_types.append("do_hard (random col = 0)")

            # Unimportant col = colors[0].
            x_hard_1 = x.copy()
            y_hard_1 = y.copy()
            not_div_5 = [j for j in range(x.shape[1]) if (j % modulo != 0) and (len(np.unique(x[:,j])) > 1)]
            random_col = np.random.choice(not_div_5, size = 1)[0]
            x_hard_1[:,random_col] = colors[0]
            y_hard_1[:,random_col] = colors[0]
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)
            cf_types.append("do_hard (random col = colors[0])")

            #########################
            # Soft interventions.
            #########################

            # Get random colors not used in original grids.
            cf_colors = [x for x in range(1,10) if x not in colors]
            random.shuffle(cf_colors)
            cf_colors = cf_colors[:3]
            for color in cf_colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==colors[0]] = color
                y_soft_color = y.copy()
                y_soft_color[y_soft_color==colors[0]] = color
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

            if plot:
                for x,y,cf_type in zip(cf_inputs,cf_outputs,cf_types):
                    if plot_type == "single":
                        print(cf_type)
                        self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                    else:
                        print(cf_type)
                        self.u.plot_input_output(input_grid = x, 
                                                 output_grid = y, 
                                                 grid = grid, 
                                                 figsize = figsize)
                    

            #assert len(cf_types) == len(cf_inputs) and len(cf_inputs) == len(cf_outputs)

            train_list.append(
                {"input": x.tolist(),
                 "output": y.tolist(), 
                 "cf_types": cf_types, 
                 "cf_inputs": [x.tolist() for x in cf_inputs],
                 "cf_outputs": [x.tolist() for x in cf_outputs]}
                             )
            
        sample_dict["train"] = train_list
        return sample_dict


    def task_SCMz750(self,
                     size: tuple = (10,10), # (15,15), (20,20)
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC task 705a3229.

        Pixels are extended into paths from top to bottom and left to right.
        That is, the path for a starting pixel in row i will be overlapped by 
        any path whose starting pixel was below row i.
        The path for a starting pixel in column j will be overlapped by any path 
        whose starting pixel was to the right of column j.
        '''

        scm = '''def SCMz750(size: tuple = (10,10)):
            x = np.random.choice(range(10), p = [0.955]+[0.005]*9, size = size)
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if x[i,j] != 0:
                        y[i,:j] = x[i,j] 
                        y[:i,j] = x[i,j]
                    else:
                        y[i,j] = x[i,j] 
            return x,y
        '''
        sample_dict = {"scm": scm}

        def SCMz750(size: tuple = (10,10), x: np.array = None):
            #_u = np.random.binomial(n = 1, p = 0.005, size = size)
            if x is None:
                x = np.random.choice(range(10), p = [0.955]+[0.005]*9, size = size)
            y = x.copy()
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if x[i,j] != 0:
                        y[i,:j] = x[i,j] 
                        y[:i,j] = x[i,j]
                    else:
                        y[i,j] = x[i,j] 
            return x,y

        # Get test sample.
        x_test,y_test = SCMz750(size = size)
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            x,y = SCMz750(size = size)
            colors = [c for c in np.unique(x) if c > 0]
            while len(colors) == 0:
                x,y = SCMz750(size = size)
                colors = [c for c in np.unique(x) if c > 0]

            if plot:
                if plot_type == "single":
                    self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                else:
                    self.u.plot_input_output(input_grid = x, output_grid = y, grid = grid, figsize = figsize)

            #########################
            # Hard interventions.
            #########################

            cf_inputs = []
            cf_outputs = []
            cf_types = []
            
            # Color = 0.
            for color in colors:
                x_hard = x.copy()
                x_hard[x_hard==color] = 0
                x_hard,y_hard = SCMz750(size = size, x = x_hard)
                cf_inputs.append(x_hard)
                cf_outputs.append(y_hard)
                cf_types.append(f"do_hard ({color} -> 0)")

            #########################
            # Soft interventions.
            #########################

            # Get random colors not used in original grids.
            cf_colors = [x for x in range(1,10) if x not in colors]
            random.shuffle(cf_colors)
            cf_colors = cf_colors[:3]
            for color in cf_colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==colors[0]] = color
                y_soft_color = y.copy()
                y_soft_color[y_soft_color==colors[0]] = color
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

            if plot:
                for x,y,cf_type in zip(cf_inputs,cf_outputs,cf_types):
                    if plot_type == "single":
                        print(cf_type)
                        self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
                    else:
                        print(cf_type)
                        self.u.plot_input_output(input_grid = x, 
                                                 output_grid = y, 
                                                 grid = grid, 
                                                 figsize = figsize)
                    

            #assert len(cf_types) == len(cf_inputs) and len(cf_inputs) == len(cf_outputs)

            train_list.append(
                {"input": x.tolist(),
                 "output": y.tolist(), 
                 "cf_types": cf_types, 
                 "cf_inputs": [x.tolist() for x in cf_inputs],
                 "cf_outputs": [x.tolist() for x in cf_outputs]}
                             )
            
        sample_dict["train"] = train_list
        return sample_dict


    def task_SCM6cjq(self,
                     size: tuple = (10,10), # (15,15), (20,20)
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:
        
        def SCM6cjq(size: tuple = (10,10), x: np.array = None):
            if x is None:
                u1 = np.random.binomial(n = 1, p = 0.2, size = size)
                u2 = np.random.binomial(n = 1, p = 0.1, size = size)
                x = 6*u1 + u2
            y = x.copy()
            orange = np.argwhere(y==7)
            for idx in orange:
                y[idx[0]+1:,idx[1]] = 4
                y[idx[0],idx[1]] = 3
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCM6cjq(size: tuple = (10,10)):
            u1 = np.random.binomial(n = 1, p = 0.2, size = size)
            u2 = np.random.binomial(n = 1, p = 0.1, size = size)
            x = 6*u1 + u2
            y = x.copy()
            orange = np.argwhere(y==7)
            for idx in orange:
                y[idx[0]+1:,idx[1]] = 4
                y[idx[0],idx[1]] = 3
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCM6cjq(size = size)
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
            x,y = SCM6cjq(size = size)

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

            # Hard interventions on colors.
            colors = np.unique(x)
            for color in colors:
                x_hard = x.copy()
                x_hard[x_hard==color] = 0
                x_hard,y_hard = SCM6cjq(size = size, x = x_hard)
                cf_inputs.append(x_hard)
                cf_outputs.append(y_hard)
                cf_types.append(f"do_hard ({color} -> 0)")
    
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
            for transform in [np.fliplr]:
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


    def task_SCMesea(self,
                     size: tuple = (10,10), # (15,15), (20,20)
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:
        
        def SCMesea(size: tuple = (10,10), x: np.array = None):
            if x is None:
                _u = np.random.binomial(n = 1, p = 0.1, size = size)*5
                x = np.random.choice(range(10), p = [0.91]+[0.01]*9, size = size)
                x = x+_u
            y = x.copy()
            for i in range(y.shape[0]):
                for j in reversed(range(y.shape[1])):
                    if y[i,j] not in [0,5]:
                        y[i,j:] = x[i,j]
                    else:
                        y[i,j] = x[i,j] 
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMesea(size: tuple = (10,10), x: np.array = None):
            if x is None:
                _u = np.random.binomial(n = 1, p = 0.1, size = size)*5
                x = np.random.choice(range(10), p = [0.91]+[0.01]*9, size = size)
                x = x+_u
            y = x.copy()
            for i in range(y.shape[0]):
                for j in reversed(range(y.shape[1])):
                    if y[i,j] not in [0,5]:
                        y[i,j:] = x[i,j]
                    else:
                        y[i,j] = x[i,j] 
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMesea(size)
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
            x,y = SCMesea(size)

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

            # Hard interventions on colors.
            colors = np.unique(x)
            for color in colors:
                x_hard = x.copy()
                x_hard[x_hard==color] = 0
                x_hard,y_hard = SCMesea(size = size, x = x_hard)
                cf_inputs.append(x_hard)
                cf_outputs.append(y_hard)
                cf_types.append(f"do_hard ({color} -> 0)")
    
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


    def task_SCMwoev(self,
                     size: tuple = (10,10), # (15,15), (20,20)
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:
        
        def SCMwoev(size: tuple = (10,10), x: np.array = None):
            if x is None:
                _u = np.random.binomial(n = 1, p = 0.05, size = size)*5
                x = np.random.choice(range(10), p = [0.955]+[0.005]*9, size = size)
                x = x+_u
            y = x.copy()
            for i in range(y.shape[0]):
                not_black = np.argwhere(y[i,:]>0)
                for idx_list in not_black:
                    for idx in idx_list:
                        if x[i,idx] != 5:
                            y[i,idx:] = x[i,idx]
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMwoev(size: tuple = (10,10), x: np.array = None):
            if x is None:
                _u = np.random.binomial(n = 1, p = 0.05, size = size)*5
                x = np.random.choice(range(10), p = [0.955]+[0.005]*9, size = size)
                x = x+_u
            y = x.copy()
            for i in range(y.shape[0]):
                not_black = np.argwhere(y[i,:]>0)
                for idx_list in not_black:
                    for idx in idx_list:
                        if x[i,idx] != 5:
                            y[i,idx:] = x[i,idx]
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMwoev(size)
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
            x,y = SCMwoev(size)

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

            # Hard interventions on colors.
            colors = np.unique(x)
            for color in colors:
                x_hard = x.copy()
                x_hard[x_hard==color] = 0
                x_hard,y_hard = SCMwoev(size = size, x = x_hard)
                cf_inputs.append(x_hard)
                cf_outputs.append(y_hard)
                cf_types.append(f"do_hard ({color} -> 0)")
    
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
