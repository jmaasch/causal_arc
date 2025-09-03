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


class TaskCounting:

    
    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()

        # Dictionary mapping CausalARC-AGI-1 tasks to the ARC-AGI-1 task that inspired them.
        # Tasks may only be loosely based on each other.
        # Tasks not inspired by a specific ARC-AGI-1 task map to None. 
        self.reference_dict = {"SCMm5ob": "f3cdc58f", 
                               "SCMev5t": "2753e76c", 
                               "SCMfuy3": None,
                               "SCM95ls": None,
                               "SCMhlh2": None,
                               "SCM43rz": None}

    def task_SCMm5ob(self,
                     size: tuple = (10,10),
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC-AGI-1 task f3cdc58f.
        '''

        # Add string rep of SCM.
        scm_str = '''def SCMm5ob(colors):
            p = [0.8]+[0.05]*4
            x = np.random.choice(colors, size = size, replace = True, p = p)
            
            counts = np.unique(x, return_counts = True)
            y = np.zeros((np.max(counts[1][1:]), 4)).astype(int)
            counts = dict(zip(counts[0], counts[1]))
            for color,count in counts.items():
                if color > 0:
                    y[-count:,colors.index(color)-1] = color
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Helper functions.
        def SCMm5ob(colors):
            p = [0.8]+[0.05]*4
            x = np.random.choice(colors, size = size, replace = True, p = p)
            
            counts = np.unique(x, return_counts = True)
            y = np.zeros((np.max(counts[1][1:]), 4)).astype(int)
            counts = dict(zip(counts[0], counts[1]))
            for color,count in counts.items():
                if color > 0:
                    y[-count:,colors.index(color)-1] = color
            return x,y
        
        # Get test sample.
        colors = [0] + np.random.choice(list(range(1,10)), size = 4, replace = False).tolist()
        x_test,y_test = SCMm5ob(colors)
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
            x,y = SCMm5ob(colors)

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
            nonzero = [x for x in colors if x > 0]
            for color in nonzero:
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
                x_color[x_color == 0] = cf_color
                y_color = y.copy()
                y_color[y_color == 0] = cf_color
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_soft (color {color} = {cf_color})")
    
            # Soft intervention on geometry.
            for transform in [np.flipud, np.fliplr, np.rot90]:
                x_transform = x.copy()
                x_transform = transform(x_transform)
                y_transform = y.copy()
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


    def task_SCMev5t(self,
                     size: tuple = (10,10),
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        Based on ARC-AGI-1 task 2753e76c.
        '''

        def SCMev5t(colors: list, size: tuple = (10,10)):
            n_per_color = np.random.choice([1,2,3], size = len(colors))
            true_n_per_color = [x for x in n_per_color]
            x = np.zeros(size).astype(int)
            for i in range(len(colors)):
                for n in range(n_per_color[i]):
                    try:
                        random_size = np.random.choice([2,3], size = 2)
                        sprite = np.ones(random_size).astype(int)*colors[i]
                        x = self.a.add_sprite(x, sprite = sprite)
                    except:
                        if true_n_per_color[i] == 0:
                            pass
                        else: 
                            true_n_per_color[i] -= 1

            sorted_vals = sorted(zip(true_n_per_color,colors))
            sorted_vals = [v for v in sorted_vals if v[0] > 0]
            y = np.zeros((len(sorted_vals), max(true_n_per_color))).astype(int)
            for i in range(len(sorted_vals)):
                y[i,:sorted_vals[i][0]] = sorted_vals[i][1]
            return x,y

        scm_str = '''def SCMev5t(colors: list, size: tuple = (10,10)):
            n_per_color = np.random.choice([1,2,3], size = len(colors))
            true_n_per_color = [x for x in n_per_color]
            x = np.zeros(size).astype(int)
            for i in range(len(colors)):
                for n in range(n_per_color[i]):
                    try:
                        random_size = np.random.choice([2,3], size = 2)
                        sprite = np.ones(random_size).astype(int)*colors[i]
                        x = self.a.add_sprite(x, sprite = sprite)
                    except:
                        if true_n_per_color[i] == 0:
                            pass
                        else: 
                            true_n_per_color[i] -= 1

            sorted_vals = sorted(zip(true_n_per_color,colors))
            sorted_vals = [x for x in sorted_vals if x[0] > 0]
            y = np.zeros((len(sorted_vals), max(true_n_per_color))).astype(int)
            for i in range(len(sorted_vals)):
                y[i,:sorted_vals[i][0]] = sorted_vals[i][1]
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        colors = np.random.choice(list(range(1,10)), size = 3, replace = False)
        x_test,y_test = SCMev5t(colors, size)
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
            x,y = SCMev5t(colors, size)

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
            for color in np.unique(x):
                x_color = x.copy()
                x_color[x_color == color] = 0
                y_color = y.copy()
                y_color[y_color == color] = 0
                rows = (y_color[:, 0] == 0)
                y_color = np.delete(y_color, (rows), axis = 0)
                cf_inputs.append(x_color)
                cf_outputs.append(y_color)
                cf_types.append(f"do_hard (color {color} = 0)")

            # Soft interventions on colors.
            colors = [c for c in np.unique(x) if c > 0]
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


    def task_SCMfuy3(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:
        
        def SCMfuy3():
            size = (3,3)
            total_colors = np.random.choice([5,6,7], size = 1)[0]
            half = total_colors//2
            colors = np.random.choice(list(range(1,10)), replace = False, size = half).tolist()
            not_in = [x for x in list(range(1,10)) if x not in colors]
            most_freq_color = np.random.choice(not_in, replace = False, size = 1)[0]
            most_freq = [most_freq_color for i in range(half)]
            colors = colors + most_freq
            random.shuffle(colors)
            u_list = [np.random.binomial(n = 1, p = 0.85, size = size) for i in range(len(colors))]
            x = [u_list[i]*colors[i] for i in range(len(colors))]
            x = [np.pad(z, pad_width = 1) for z in x]
            x = np.concatenate(x, axis = 1)
            y = [np.pad(np.ones((1,1)),pad_width=1)*colors[i] if colors[i] != most_freq_color else np.ones(size)*most_freq_color for i in range(len(colors))]
            y = np.concatenate(y, axis = 0)
            return x,y

        scm = '''def SCMfuy3():
            size = (3,3)
            half = total_colors//2
            colors = np.random.choice(list(range(1,10)), replace = False, size = half).tolist()
            not_in = [x for x in list(range(1,10)) if x not in colors]
            most_freq_color = np.random.choice(not_in, replace = False, size = 1)[0]
            most_freq = [most_freq_color for i in range(half)]
            colors = colors + most_freq
            random.shuffle(colors)
            u_list = [np.random.binomial(n = 1, p = 0.85, size = size) for i in range(len(colors))]
            x = [u_list[i]*colors[i] for i in range(len(colors))]
            x = [np.pad(z, pad_width = 1) for z in x]
            x = np.concatenate(x, axis = 1)
            y = [np.pad(np.ones((1,1)),pad_width=1)*colors[i] if colors[i] != most_freq_color else np.ones(size)*most_freq_color for i in range(len(colors))]
            y = np.concatenate(y, axis = 0)
            return x,y
        '''
        sample_dict = {"scm": scm}

        # Get test sample.
        x_test,y_test = SCMfuy3()
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
            x,y = SCMfuy3()

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
            for transform in [np.flipud, np.fliplr]:
                x_transform = transform(x.copy())
                y_transform = y.copy()
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
        

    def task_SCMhlh2(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        def SCMhlh2():
            size = np.random.choice(range(3,10), size = 2)
            colors = np.random.choice(range(2,10), size = 2, replace = False)
            _u = np.random.binomial(n = 1, p = 0.8, size = size)
            x = _u.copy()
            count = x[0,0]+x[-1,0]+x[0,-1]+x[-1,-1]
            if count == 0:
                x[0,0] = 1
            x[0,0] = x[0,0]*colors[0]
            x[-1,0] = x[-1,0]*colors[0]
            x[0,-1] = x[0,-1]*colors[0]
            x[-1,-1] = x[-1,-1]*colors[0]
            x[x.shape[0]//2,x.shape[1]//2] = colors[1]
            y = np.ones((1,count)).astype(int)*colors[1]
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCMhlh2():
            size = np.random.choice(range(3,10), size = 2)
            colors = np.random.choice(range(2,10), size = 2, replace = False)
            _u = np.random.binomial(n = 1, p = 0.8, size = size)
            x = _u.copy()
            count = x[0,0]+x[-1,0]+x[0,-1]+x[-1,-1]
            x[0,0] = x[0,0]*colors[0]
            x[-1,0] = x[-1,0]*colors[0]
            x[0,-1] = x[0,-1]*colors[0]
            x[-1,-1] = x[-1,-1]*colors[0]
            x[x.shape[0]//2,x.shape[1]//2] = colors[1]
            y = np.ones((1,count)).astype(int)*colors[1]
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCMhlh2()
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
            x,y = SCMhlh2()

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
                y_transform = y.copy()
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


    def task_SCM43rz(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        def SCM43rz():
            colors = np.random.choice([1,2,3,4], size = 2, replace = False)
            u1 = np.random.binomial(n = 1, size = (5,5), p = 0.5)
            u2 = np.random.binomial(n = 1, size = (5,5), p = 0.5)
            x = u1*colors[0] + u2*colors[1]
            x = np.pad(x, pad_width = 2)
            counts = np.unique(x, return_counts = True)
            # Getting sorted indices.
            idx = np.argsort(counts[1]) 
            # Sorting both lists.
            colors = [counts[0][i] for i in idx] 
            counts = [counts[1][i] for i in idx] 
            least_freq = [colors[i] for i in range(len(colors)) if counts[i] == min(counts)]
            y = np.array(least_freq).reshape(-1,1)
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCM43rz():
            colors = np.random.choice([1,2,3,4], size = 2, replace = False)
            u1 = np.random.binomial(n = 1, size = (5,5), p = 0.5)
            u2 = np.random.binomial(n = 1, size = (5,5), p = 0.5)
            x = u1*colors[0] + u2*colors[1]
            x = np.pad(x, pad_width = 2)
            counts = np.unique(x, return_counts = True)
            # Getting sorted indices.
            idx = np.argsort(counts[1]) 
            # Sorting both lists.
            colors = [counts[0][i] for i in idx] 
            counts = [counts[1][i] for i in idx] 
            least_freq = [colors[i] for i in range(len(colors)) if counts[i] == min(counts)]
            y = np.array(least_freq).reshape(-1,1)
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCM43rz()
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
            x,y = SCM43rz()

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
                y_transform = y.copy()
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


    def task_SCM95ls(self,
                     n_examples: int = 5,
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        # Extend sprites by total number of sprites.
        def SCM95ls():
            sprites = []
            n = np.random.choice([2,3,4,5], size = 1)[0]
            colors = np.random.choice([1,2,3,4,6,7,8,9], size = n, replace = False)
            for i in range(n):
                sprite = np.random.binomial(n = 1, size = (3,3), p = 0.8)*colors[i]
                sprite = np.pad(sprite, pad_width = 1)
                sprites.append(sprite)
            x = np.concatenate(sprites, axis = 1)
            y = np.concatenate([np.repeat(s,n,axis=1) for s in sprites], axis = 1)
            return x,y

        # Get SCM string representation.
        scm_str = '''def SCM95ls():
            sprites = []
            n = np.random.choice([2,3,4], size = 1)[0]
            colors = np.random.choice([1,2,3,4,6,7,8,9], size = n, replace = False)
            for i in range(n):
                sprite = np.random.binomial(n = 1, size = (3,3), p = 0.8)*colors[i]
                sprite = np.pad(sprite, pad_width = 1)
                sprites.append(sprite)
            x = np.concatenate(sprites, axis = 1)
            y = np.concatenate([np.repeat(s,n,axis=1) for s in sprites], axis = 1)
            return x,y
        '''
        sample_dict = {"scm": scm_str}

        # Get test sample.
        x_test,y_test = SCM95ls()
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
            x,y = SCM95ls()

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
