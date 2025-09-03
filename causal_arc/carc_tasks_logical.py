# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import math
import random
import networkx as nx

# Custom modules.
from carc_utils import UtilsARC
from carc_augment import AugmentARC


class TaskLogical:
    

    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()

        # Dictionary mapping CausalARC tasks to the ARC task that inspired them.
        # Tasks may only be loosely based on each other.
        # Tasks not inspired by a specific ARC task map to None. 
        self.reference_dict = {"SCMdky5": "31d5ba1a", 
                               "SCMu3am": "31d5ba1a", 
                               "SCMtcbq": "31d5ba1a"}

    
    def task_31d5ba1a(self,
                      n: int = 5, # total samples
                      plot: bool = False,
                      plot_type: str = "input_output", # "single"
                      figsize: tuple = (4,3),
                      grid: bool = True,
                      interventions: list = None, # ["color", "hard", "fun"]
                      colors: list = [1,3,7] # Anything in range(0,10)
                      ) -> np.array:

        '''
        SCM recovered for ARC-AGI-1 task 31d5ba1a.
        Logical xor causal functions.
        '''

        samples = []
        for i in range(n):
            sample = self.task_31d5ba1a_single(plot = plot,
                                               plot_type = plot_type,
                                               figsize = figsize,
                                               grid = grid,
                                               interventions = interventions,
                                               colors = colors)
            samples.append(sample)
        return samples
        

    def task_31d5ba1a_single(self,
                             plot: bool = False,
                             plot_type: str = "input_output", # "single"
                             figsize: tuple = (4,3),
                             grid: bool = True,
                             interventions: list = None, # ["color", "hard", "fun"]
                             colors: list = [1,3,7] # Anything in range(0,10)
                             ) -> np.array:

        '''
        SCM recovered for ARC-AGI-1 task 31d5ba1a.
        '''

        # Sample input grid.
        brown = np.random.choice([0,9], size = (3,5))
        yellow = np.random.choice([0,4], size = (3,5))
        x = np.concatenate((brown,yellow), axis = 0).astype(int)

        # Construct output grid.
        y = np.logical_xor(brown,yellow).astype(int)
        y *= 6
        sample_dict = {"Sample": (x,y)}
    
        if plot:
            print("Original sample")
            if plot_type == "single":
                self.u.plot_single(input_grid = x, grid = grid, figsize = figsize)
                self.u.plot_single(input_grid = y, grid = grid, figsize = figsize)
            else:
                self.u.plot_input_output(input_grid = x, output_grid = y, grid = grid, figsize = figsize)

        # Construct interventional samples.
        if interventions is not None:
            if "hard" in interventions:
                brown_hard_0 = np.zeros((3,5)).astype(int)
                x_hard_0 = np.concatenate((brown_hard_0,yellow), axis = 0).astype(int)
                y_hard_0 = np.logical_xor(brown_hard_0,yellow).astype(int)*6
            
                brown_hard_1 = np.ones((3,5)).astype(int)*9
                x_hard_1 = np.concatenate((brown_hard_1,yellow), axis = 0).astype(int)
                y_hard_1 = np.logical_xor(brown_hard_1,yellow).astype(int)*6

                sample_dict["do_hard (0)"] = (x_hard_0,y_hard_0)
                sample_dict["do_hard (1)"] = (x_hard_1,y_hard_1)

                if plot:
                    if plot_type == "single":
                        print("Hard intervention: x_brown = 0")
                        self.u.plot_single(input_grid = x_hard_0, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_hard_0, grid = grid, figsize = figsize)
                        print("Hard intervention: x_brown = 1")
                        self.u.plot_single(input_grid = x_hard_1, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_hard_1, grid = grid, figsize = figsize)
                    else:
                        print("Hard intervention: x_brown = 0")
                        self.u.plot_input_output(input_grid = x_hard_0, output_grid = y_hard_0, grid = grid, figsize = figsize)
                        print("Hard intervention: x_brown = 1")
                        self.u.plot_input_output(input_grid = x_hard_1, output_grid = y_hard_1, grid = grid, figsize = figsize)
                
            if "color" in interventions:
                for color in colors:
                    brown_soft_color = brown.copy()
                    brown_soft_color[brown_soft_color==9] = color
                    x_soft_color = np.concatenate((brown_soft_color,yellow), axis = 0).astype(int)
                    y_soft_color = np.logical_xor(brown_soft_color,yellow).astype(int)*6

                    sample_dict[f"do_soft (color = {color})"] = (x_soft_color,y_soft_color)

                    if plot:
                        if plot_type == "single":
                            print("Soft intervention: color")
                            self.u.plot_single(input_grid = x_soft_color, grid = grid, figsize = figsize)
                            self.u.plot_single(input_grid = y_soft_color, grid = grid, figsize = figsize)
                        else:
                            print("Soft intervention: color")
                            self.u.plot_input_output(input_grid = x_soft_color, 
                                                output_grid = y_soft_color, 
                                                grid = grid, 
                                                figsize = figsize)
            if "fun" in interventions:
                # Change local function for x[i,j] to include logical not.
                brown_soft_fun = brown.copy()
                brown_soft_fun[brown_soft_fun==9] = 1
                brown_soft_fun = np.logical_not(brown_soft_fun).astype(int)*9
                x_soft_fun = np.concatenate((brown_soft_fun,yellow), axis = 0).astype(int)
                y_soft_fun = np.logical_xor(brown_soft_fun,yellow).astype(int)*6

                sample_dict["do_soft (fun)"] = (x_soft_fun,y_soft_fun)

                if plot:
                    if plot_type == "single":
                        print("Soft intervention: causal function")
                        self.u.plot_single(input_grid = x_soft_fun, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_soft_fun, grid = grid, figsize = figsize)
                    else:
                        print("Soft intervention: causal function")
                        self.u.plot_input_output(input_grid = x_soft_fun, 
                                            output_grid = y_soft_fun, 
                                            grid = grid, 
                                            figsize = figsize)
            
        return sample_dict


    def get_fun(self, fun: str):

        '''
        fun: str in ["or", "xor", "and"]

        fun_dict = {"and": np.logical_and, 
                    "or": np.logical_or, 
                    "xor": np.logical_xor}
        '''
        
        if fun == "or":
            fun = np.logical_or
            fun_str = "np.logical_or"
        elif fun == "and":
            fun = np.logical_and
            fun_str = "np.logical_and"
        elif fun == "xor":
            fun = np.logical_xor
            fun_str = "np.logical_xor"
        else:
            raise ValueError("Invalid input: fun must be in ['or', 'and', 'xor'].")
        return fun,fun_str


    def get_causal_graph(parent_dict: dict) -> nx.DiGraph:

        #if sparse:
        #    return nx.from_scipy_sparse_array(parent_dict, create_using = nx.DiGraph)
        return nx.from_dict_of_lists(parent_dict, create_using = nx.DiGraph)

    
    def task_SCMdky5(self,
                     fun: str = "and", # "or", "xor", "and"
                     upper_color: int = 2,
                     lower_color: int = 6,
                     output_color: int = 7,
                     size: tuple = (4,4),
                     n_examples: int = 5, # Total demonstration pairs.
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        CausalARC task SCMdky5.
        '''
    
        # Assign causal function.
        fun,fun_str = self.get_fun(fun)

        # Get str representation of SCM.
        scm_str = '''def SCMdky5():
            upper = np.random.choice([0,upper_color], size = size)
            lower = np.random.choice([0,lower_color], size = size)
            x = np.concatenate((upper,lower), axis = 0).astype(int)
            y = fun(upper,lower).astype(int)
            y *= output_color
            return x,y,(upper,lower)
        '''
        sample_dict = {"scm": scm_str}

        # Get LaTeX math representation of SCM.
        latex = r'''\begin{align}
            \mathbf{u}[i,j] &\sim \mathrm{Ber}(0.5) & \text{for} \; i \in [0,5], j \in [0,4] \\
            \mathbf{x}[i,j] &= \begin{cases} 9 \cdot \mathbf{u}[i,j] \quad \text{if} \; i < 3 \\ 4 \cdot \mathbf{u}[i,j] \quad \text{else} \end{cases} & \text{for} \; i \in [0,5], j \in [0,4] \\
            \mathbf{y}[i,j] &= {6 \cdot \mathrm{xor} \left( \mathbf{x}[i,j], \; \mathbf{x}[i+3,j] \right) & \text{for} \; i \in [0,2], j \in [0,4].
        \end{align}'''
        sample_dict["latex"] = latex

        # Helper function.
        def SCMdky5():
            upper = np.random.choice([0,upper_color], size = size)
            lower = np.random.choice([0,lower_color], size = size)
            x = np.concatenate((upper,lower), axis = 0).astype(int)
            y = fun(upper,lower).astype(int)
            y *= output_color
            return x,y,(upper,lower)

        # Get test sample.
        x_test,y_test,ul_test = SCMdky5()
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]

        # Get adjacency matrix.
        parent_dict = dict()
        for i in range(y_test.shape[0]):
            for j in range(y_test.shape[1]):
                parent_dict[f"x_{i}_{j}"] = [f"y_{i}_{j}"]
                parent_dict[f"x_{i+y_test.shape[0]}_{j}"] = [f"y_{i}_{j}"]
        g_true = nx.from_dict_of_lists(parent_dict, create_using = nx.DiGraph)
        a_true = nx.to_numpy_array(g_true, nodelist = sorted(g_true.nodes())).astype(int)
        #a_true = nx.to_scipy_sparse_array(g_true, nodelist = sorted(g_true.nodes())).astype(int)
        sample_dict["parent_to_children_dict"] = parent_dict
        sample_dict["directed_adjacency_matrix"] = a_true.tolist() #tolil() if scipy sparse
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            x,y,ul = SCMdky5()
            upper = ul[0]
            lower = ul[1]

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
            cf_types = ["do_hard upper (0)", 
                        "do_hard upper (1)", 
                        "do_hard lower (0)", 
                        "do_hard lower (1)",  
                        "do_soft (fun)"]
            
            # Intervene on upper.
            upper_hard_0 = np.zeros(size).astype(int)
            x_hard_0 = np.concatenate((upper_hard_0,lower), axis = 0).astype(int)
            y_hard_0 = fun(upper_hard_0,lower).astype(int)*output_color
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            upper_hard_1 = np.ones(size).astype(int)*upper_color
            x_hard_1 = np.concatenate((upper_hard_1,lower), axis = 0).astype(int)
            y_hard_1 = fun(upper_hard_1,lower).astype(int)*output_color
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            # Intervene on lower.
            lower_hard_0 = np.zeros(size).astype(int)
            x_hard_0 = np.concatenate((upper,lower_hard_0), axis = 0).astype(int)
            y_hard_0 = fun(upper,lower_hard_0).astype(int)*output_color
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            lower_hard_1 = np.ones(size).astype(int)*lower_color
            x_hard_1 = np.concatenate((upper,lower_hard_1), axis = 0).astype(int)
            y_hard_1 = fun(upper,lower_hard_1).astype(int)*output_color
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            if plot:
                if plot_type == "single":
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_0, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_0, grid = grid, figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_1, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_1, grid = grid, figsize = figsize)
                else:
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_0, 
                                             output_grid = y_hard_0, 
                                             grid = grid, 
                                             figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_1, 
                                             output_grid = y_hard_1, 
                                             grid = grid, 
                                             figsize = figsize)
            #########################
            # Soft interventions.
            #########################

            # Change local function for x[i,j] to include logical not.
            upper_soft_fun = upper.copy()
            upper_soft_fun[upper_soft_fun==upper_color] = 1
            upper_soft_fun = np.logical_not(upper_soft_fun).astype(int)*upper_color
            x_soft_fun = np.concatenate((upper_soft_fun,lower), axis = 0).astype(int)
            y_soft_fun = fun(upper_soft_fun,lower).astype(int)*output_color
            cf_inputs.append(x_soft_fun)
            cf_outputs.append(y_soft_fun)

            if plot:
                if plot_type == "single":
                    print("Soft intervention: causal function")
                    self.u.plot_single(input_grid = x_soft_fun, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_soft_fun, grid = grid, figsize = figsize)
                else:
                    print("Soft intervention: causal function")
                    self.u.plot_input_output(input_grid = x_soft_fun, 
                                             output_grid = y_soft_fun, 
                                             grid = grid, 
                                             figsize = figsize)

            # Get random colors not used in original grids.
            colors = [x for x in range(1,10) if x not in [upper_color,lower_color,output_color]]
            random.shuffle(colors)
            colors = colors[:3]
            for color in colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==upper_color] = color
                y_soft_color = y.copy()
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

                if plot:
                    if plot_type == "single":
                        print("Soft intervention: color")
                        self.u.plot_single(input_grid = x_soft_color, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_soft_color, grid = grid, figsize = figsize)
                    else:
                        print("Soft intervention: color")
                        self.u.plot_input_output(input_grid = x_soft_color, 
                                            output_grid = y_soft_color, 
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


    def task_SCMu3am(self,
                     size: tuple = (4,4), 
                     fun_0: str = "and",
                     fun_1: str = "or",
                     alternate_axis: int = 1, # 1 for columns, 0 for rows.
                     upper_color: int = 1, 
                     lower_color: int = 2,
                     output_color: int = 3,
                     n_examples: int = 5, # Total demonstration pairs.
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        CausalARC task SCMu3am.

        Alternating causal functions: 
        Every even column (or row) uses logical or, 
        while odd columns (or rows) use logical and.
        '''

        # Assign causal function.
        fun_0,fun_0_str = self.get_fun(fun_0)
        fun_1,fun_1_str = self.get_fun(fun_1)

        # Get str representation of SCM.
        scm_str = '''def SCMu3am(upper: np.array = None, lower: np.array = None):
            # Used when there is no intervention.
            if upper is None:
                upper = np.random.choice([0,upper_color], size = size)
            if lower is None:
                lower = np.random.choice([0,lower_color], size = size)

            # Get x and y.
            x = np.concatenate((upper,lower), axis = 0).astype(int)
            y = np.zeros(size).astype(int)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if alternate_axis:
                        if j % 2 == 0:
                            y[i,j] = fun_0(upper[i,j],lower[i,j]).astype(int)
                        else:
                            y[i,j] = fun_1(upper[i,j],lower[i,j]).astype(int)
                    else:
                        if i % 2 == 0:
                            y[i,j] = fun_0(upper[i,j],lower[i,j]).astype(int)
                        else:
                            y[i,j] = fun_1(upper[i,j],lower[i,j]).astype(int)
            y *= output_color
            return x,y,(upper,lower)
        '''
        sample_dict = {"scm": scm_str}

        # Get LaTeX math representation of SCM.
        latex = r'''\begin{align}
            \mathbf{u}[i,j] &\sim \mathrm{Ber}(0.5) & \text{for} \; i \in [0,5], j \in [0,4] \\
            \mathbf{x}[i,j] &= \begin{cases} 9 \cdot \mathbf{u}[i,j] \quad \text{if} \; i < 3 \\ 4 \cdot \mathbf{u}[i,j] \quad \text{else} \end{cases} & \text{for} \; i \in [0,5], j \in [0,4] \\
            \mathbf{y}[i,j] &= {6 \cdot \mathrm{xor} \left( \mathbf{x}[i,j], \; \mathbf{x}[i+3,j] \right) & \text{for} \; i \in [0,2], j \in [0,4].
        \end{align}'''
        sample_dict["latex"] = latex

        # Helper function.
        def SCMu3am(upper: np.array = None, lower: np.array = None):
            # Used when there is no intervention.
            if upper is None:
                upper = np.random.choice([0,upper_color], size = size)
            if lower is None:
                lower = np.random.choice([0,lower_color], size = size)

            # Get x and y.
            x = np.concatenate((upper,lower), axis = 0).astype(int)
            y = np.zeros(size).astype(int)
            for i in range(y.shape[0]):
                for j in range(y.shape[1]):
                    if alternate_axis:
                        if j % 2 == 0:
                            y[i,j] = fun_0(upper[i,j],lower[i,j]).astype(int)
                        else:
                            y[i,j] = fun_1(upper[i,j],lower[i,j]).astype(int)
                    else:
                        if i % 2 == 0:
                            y[i,j] = fun_0(upper[i,j],lower[i,j]).astype(int)
                        else:
                            y[i,j] = fun_1(upper[i,j],lower[i,j]).astype(int)
            y *= output_color
            return x,y,(upper,lower)

        # Get test sample.
        x_test,y_test,ul_test = SCMu3am()
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]

        # Get adjacency matrix.
        parent_dict = dict()
        for i in range(y_test.shape[0]):
            for j in range(y_test.shape[1]):
                parent_dict[f"x_{i}_{j}"] = [f"y_{i}_{j}"]
                parent_dict[f"x_{i+y_test.shape[0]}_{j}"] = [f"y_{i}_{j}"]
        g_true = nx.from_dict_of_lists(parent_dict, create_using = nx.DiGraph)
        a_true = nx.to_numpy_array(g_true, nodelist = sorted(g_true.nodes())).astype(int)
        #a_true = nx.to_scipy_sparse_array(g_true, nodelist = sorted(g_true.nodes())).astype(int)
        sample_dict["parent_to_children_dict"] = parent_dict
        sample_dict["directed_adjacency_matrix"] = a_true.tolist() #tolil() if scipy sparse
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            x,y,ul = SCMu3am()
            upper = ul[0]
            lower = ul[1]

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
            cf_types = ["do_hard upper (0)", 
                        "do_hard upper (1)", 
                        "do_hard lower (0)", 
                        "do_hard lower (1)",  
                        "do_soft (fun)"]
            
            # Intervene on upper.
            upper_hard_0 = np.zeros(size).astype(int)
            x_hard_0,y_hard_0,_ = SCMu3am(upper = upper_hard_0)
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            upper_hard_1 = np.ones(size).astype(int)*upper_color
            x_hard_1,y_hard_1,_ = SCMu3am(upper = upper_hard_1)
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            # Intervene on lower.
            lower_hard_0 = np.zeros(size).astype(int)
            x_hard_0,y_hard_0,_ = SCMu3am(lower = lower_hard_0)
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            lower_hard_1 = np.ones(size).astype(int)*lower_color
            x_hard_1,y_hard_1,_ = SCMu3am(lower = lower_hard_1)
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            if plot:
                if plot_type == "single":
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_0, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_0, grid = grid, figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_1, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_1, grid = grid, figsize = figsize)
                else:
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_0, 
                                             output_grid = y_hard_0, 
                                             grid = grid, 
                                             figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_1, 
                                             output_grid = y_hard_1, 
                                             grid = grid, 
                                             figsize = figsize)
            #########################
            # Soft interventions.
            #########################

            # Change local function for x[i,j] to include logical not.
            upper_soft_fun = upper.copy()
            upper_soft_fun[upper_soft_fun==upper_color] = 1
            upper_soft_fun = np.logical_not(upper_soft_fun).astype(int)*upper_color
            x_soft_fun,y_soft_fun,_ = SCMu3am(upper = upper_soft_fun)
            cf_inputs.append(x_soft_fun)
            cf_outputs.append(y_soft_fun)

            if plot:
                if plot_type == "single":
                    print("Soft intervention: causal function")
                    self.u.plot_single(input_grid = x_soft_fun, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_soft_fun, grid = grid, figsize = figsize)
                else:
                    print("Soft intervention: causal function")
                    self.u.plot_input_output(input_grid = x_soft_fun, 
                                             output_grid = y_soft_fun, 
                                             grid = grid, 
                                             figsize = figsize)

            # Get random colors not used in original grids.
            colors = [x for x in range(1,10) if x not in [upper_color,lower_color,output_color]]
            random.shuffle(colors)
            colors = colors[:3]
            for color in colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==upper_color] = color
                y_soft_color = y.copy()
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

                if plot:
                    if plot_type == "single":
                        print("Soft intervention: color")
                        self.u.plot_single(input_grid = x_soft_color, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_soft_color, grid = grid, figsize = figsize)
                    else:
                        print("Soft intervention: color")
                        self.u.plot_input_output(input_grid = x_soft_color, 
                                            output_grid = y_soft_color, 
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


    def task_SCMtcbq(self,
                     size: tuple = (4,4), 
                     fun_0: str = "xor",
                     fun_1: str = "or",
                     upper_color: int = 1, 
                     middle_color: int = 4,
                     lower_color: int = 2,
                     output_color: int = 3,
                     n_examples: int = 5, # Total demonstration pairs.
                     plot: bool = False,
                     plot_type: str = "input_output", # "single"
                     figsize: tuple = (4,3),
                     grid: bool = True,
                     ) -> dict:

        '''
        CausalARC task SCMtcbq.

        Composing causal functions: 
        One logical operator operates on upper and middle subarray,
        while a second logical operator operates on this output and 
        the lower subarray.
        '''

        # Assign causal function.
        fun_0,fun_0_str = self.get_fun(fun_0)
        fun_1,fun_1_str = self.get_fun(fun_1)

        # Get str representation of SCM.
        scm_str = '''def SCMtcbq(upper: np.array = None, middle: np.array = None, lower: np.array = None):
            # Used when there is no intervention.
            if upper is None:
                upper = np.random.choice([0,upper_color], size = size)
            if middle is None:
                middle = np.random.choice([0,middle_color], size = size)
            if lower is None:
                lower = np.random.choice([0,lower_color], size = size)
            # Get x.
            x = np.concatenate((upper,middle,lower), axis = 0).astype(int)
            # Get y.
            y = fun_0(upper,middle).astype(int)
            y = fun_1(y,lower).astype(int)
            y *= output_color
            return x,y,(upper,middle,lower)
        '''
        sample_dict = {"scm": scm_str}

        # Helper function.
        def SCMtcbq(upper: np.array = None, middle: np.array = None, lower: np.array = None):
            # Used when there is no intervention.
            if upper is None:
                upper = np.random.choice([0,upper_color], size = size)
            if middle is None:
                middle = np.random.choice([0,middle_color], size = size)
            if lower is None:
                lower = np.random.choice([0,lower_color], size = size)
            # Get x.
            x = np.concatenate((upper,middle,lower), axis = 0).astype(int)
            # Get y.
            y = fun_0(upper,middle).astype(int)
            y = fun_1(y,lower).astype(int)
            y *= output_color
            return x,y,(upper,middle,lower)

        # Get test sample.
        x_test,y_test,uml_test = SCMtcbq()
        sample_dict["test"] = [{"input": x_test.tolist(), "output": y_test.tolist()}]
        
        # Get additional demonstration pairs.
        train_list = []
        for i in range(n_examples):
            x,y,uml = SCMtcbq()
            upper = uml[0]
            middle = uml[1]
            lower = uml[2]

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
            cf_types = ["do_hard upper (0)", 
                        "do_hard upper (1)", 
                        "do_hard lower (0)", 
                        "do_hard lower (1)",  
                        "do_soft (fun)"]
            
            # Intervene on upper.
            upper_hard_0 = np.zeros(size).astype(int)
            x_hard_0,y_hard_0,_ = SCMtcbq(upper = upper_hard_0, middle = middle, lower = lower)
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            upper_hard_1 = np.ones(size).astype(int)*upper_color
            x_hard_1,y_hard_1,_ = SCMtcbq(upper = upper_hard_1, middle = middle, lower = lower)
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            # Intervene on lower.
            lower_hard_0 = np.zeros(size).astype(int)
            x_hard_0,y_hard_0,_ = SCMtcbq(upper = upper, middle = middle, lower = lower_hard_0)
            cf_inputs.append(x_hard_0)
            cf_outputs.append(y_hard_0)
        
            lower_hard_1 = np.ones(size).astype(int)*lower_color
            x_hard_1,y_hard_1,_ = SCMtcbq(upper = upper, middle = middle, lower = lower_hard_1)
            cf_inputs.append(x_hard_1)
            cf_outputs.append(y_hard_1)

            if plot:
                if plot_type == "single":
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_0, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_0, grid = grid, figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_single(input_grid = x_hard_1, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_hard_1, grid = grid, figsize = figsize)
                else:
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_0, 
                                             output_grid = y_hard_0, 
                                             grid = grid, 
                                             figsize = figsize)
                    print("Hard intervention")
                    self.u.plot_input_output(input_grid = x_hard_1, 
                                             output_grid = y_hard_1, 
                                             grid = grid, 
                                             figsize = figsize)
            #########################
            # Soft interventions.
            #########################

            # Change local function for x[i,j] to include logical not in middle subarray.
            middle_soft_fun = middle.copy()
            middle_soft_fun[middle_soft_fun==middle_color] = 1
            middle_soft_fun = np.logical_not(middle_soft_fun).astype(int)*middle_color
            x_soft_fun,y_soft_fun,_ = SCMtcbq(upper = upper, middle = middle_soft_fun, lower = lower)
            cf_inputs.append(x_soft_fun)
            cf_outputs.append(y_soft_fun)

            if plot:
                if plot_type == "single":
                    print("Soft intervention: causal function")
                    self.u.plot_single(input_grid = x_soft_fun, grid = grid, figsize = figsize)
                    self.u.plot_single(input_grid = y_soft_fun, grid = grid, figsize = figsize)
                else:
                    print("Soft intervention: causal function")
                    self.u.plot_input_output(input_grid = x_soft_fun, 
                                             output_grid = y_soft_fun, 
                                             grid = grid, 
                                             figsize = figsize)

            # Get random colors not used in original grids.
            colors = [x for x in range(1,10) if x not in [upper_color,middle_color,lower_color,output_color]]
            random.shuffle(colors)
            colors = colors[:3]
            for color in colors:
                x_soft_color = x.copy()
                x_soft_color[x_soft_color==middle_color] = color
                y_soft_color = y.copy()
                cf_inputs.append(x_soft_color)
                cf_outputs.append(y_soft_color)
                cf_types.append(f"do_soft (color = {color})")

                if plot:
                    if plot_type == "single":
                        print("Soft intervention: color")
                        self.u.plot_single(input_grid = x_soft_color, grid = grid, figsize = figsize)
                        self.u.plot_single(input_grid = y_soft_color, grid = grid, figsize = figsize)
                    else:
                        print("Soft intervention: color")
                        self.u.plot_input_output(input_grid = x_soft_color, 
                                            output_grid = y_soft_color, 
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






        