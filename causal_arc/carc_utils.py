# General importations.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import json
import random
import string


class UtilsARC:
    

    def __init__(self, 
                 task_dict: dict = None, 
                 solution_dict: dict = None, 
                 submission_dict: dict = None):

        # Initialize data structures.
        self.task_dict = task_dict
        self.solution_dict = solution_dict
        self.submission_dict = submission_dict
        
        # Define color map and normalization for grid visualizations.
        # 0:black, 1:blue, 2:red, 3:green, 4:yellow
        # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown.
        self.color_map = {"black": 0,
                          "blue": 1, 
                          "red": 2, 
                          "green": 3, 
                          "yellow": 4, 
                          "gray": 5,
                          "magenta": 6, 
                          "orange": 7,
                          "sky": 8, 
                          "brown": 9}
                          #"white": 10}
        self.cmap = colors.ListedColormap(['#000000', '#0074D9', 
                                           '#FF4136', '#2ECC40', 
                                           '#FFDC00', '#AAAAAA', 
                                           '#F012BE', '#FF851B', 
                                           '#7FDBFF', '#870C25'])
                                           #'#FFFFFF'])
        self.norm = colors.Normalize(vmin = 0, vmax = 9) #vmax = 10

        # Define letter map for encoding matrices.
        # This is to avoid risk that the model interprets discrete
        # matrix values as ordinal integers.
        self.letter_map = {0: "A", #"black"
                           1: "B", #"blue" 
                           2: "C", #"red"
                           3: "D", #"green"
                           4: "E", #"yellow"
                           5: "F", #"gray"
                           6: "G", #"magenta"
                           7: "H", #"orange"
                           8: "I", #"sky"
                           9: "J"} #"brown"
                           #10: "K"} #"white"

    
    # -*-------------- Getters --------------*- 


    def get_attempts(self,
                     task_id: str):

        if None in [self.task_dict,self.solution_dict,self.submission_dict]:
            raise TypeError("Variables in [self.task_dict, self.solution_dict, self.submission_dict] cannot be None")
    
        task = self.task_dict.get(task_id)
        solution = self.solution_dict.get(task_id)[0]
        pred = self.submission_dict.get(task_id)[0]
    
        in_true = task.get("test")[0].get("input")
        out_pred_1 = pred.get("attempt_1")
        out_pred_2 = pred.get("attempt_2")
    
        return np.array(in_true), np.array(solution), np.array(out_pred_1), np.array(out_pred_2)

    
    def get_test_pair_with_demos(self,
                                 task_id: str):

        '''
        Get the input test instance, its true solution, and 
        input-output demonstration pairs.
        '''

        if None in [self.task_dict,self.solution_dict]:
            raise TypeError("Variables in [self.task_dict, self.solution_dict, self.submission_dict] cannot be None")
    
        task = self.task_dict.get(task_id)
        solution = self.solution_dict.get(task_id)[0]
    
        train = task.get("train")
        test = task.get("test")[0]
        input_test = task.get("test")[0].get("input")
    
        demo_inputs = []
        demo_outputs = []
        for pair in train:
            demo_inputs.append(np.array(pair.get("input")))
            demo_outputs.append(np.array(pair.get("output")))
    
        return np.array(input_test), np.array(solution), demo_inputs, demo_outputs
        

    # -*-------------- Visualization methods --------------*- 
    

    def plot_color_map(self):
        
        plt.figure(figsize = (3, 1), dpi = 150)
        plt.imshow([list(range(10))], cmap = self.cmap, norm = self.norm)
        plt.xticks(list(range(10)))
        plt.yticks([])
        plt.tick_params(axis = 'x', color = 'r', length = 0, grid_color = 'none')
        plt.show()
        plt.close()

    
    def plot_input_output(self,
                          input_grid: np.array, 
                          output_grid: np.array,
                          grid: bool = False,
                          figsize: tuple = (5,2),
                          subplot_titles: list = ["Input", "Output"]):

        if not grid:
            linewidth = 0
        else:
            linewidth = 0.5
            
        fig,ax = plt.subplots(1,2)
        sns.heatmap(input_grid, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[0])
        ax[0].set_title(subplot_titles[0], fontsize = 12, fontweight = 'bold')
        sns.heatmap(output_grid, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[1])
        ax[1].set_title(subplot_titles[1], fontsize = 12, fontweight = 'bold')
        fig.set_size_inches(figsize[0], figsize[1])
        plt.show()
        plt.close()


    def plot_single(self,
                    input_grid: np.array,
                    grid: bool = False,
                    title: str = None,
                    figsize: tuple = (2,2)):

        if not grid:
            linewidth = 0
        else:
            linewidth = 0.5
            
        fig,ax = plt.subplots(1,1)
        fig.set_size_inches(figsize[0],figsize[1])
        sns.heatmap(input_grid, 
                    linewidths = linewidth, 
                    cmap = self.cmap, 
                    norm = self.norm)

        if title is not None:
            plt.title(title, 
                      loc = 'center',
                      fontsize = 16, 
                      fontweight = 'bold',
                      pad = 10)

        #plt.figure(figsize = (figsize[0], figsize[1]))
        plt.show()
        plt.close()


    def plot_multi(self,
                   grid_list: list,
                   grid: bool = False,
                   title: str = None):

        if not grid:
            linewidth = 0
        else:
            linewidth = 0.5
            
        fig,ax = plt.subplots(1,len(grid_list))
        fig.set_size_inches(len(grid_list)*3,2)
        for i in range(len(grid_list)):
            sns.heatmap(grid_list[i], 
                        linewidths = linewidth, 
                        cmap = self.cmap, 
                        norm = self.norm, 
                        ax = ax[i])

        if title is not None:
            plt.suptitle(title, 
                         fontsize = 16, 
                         fontweight = 'bold', 
                         y = 1.05, # padding between title and plot
                         color = 'black')
        plt.show()
        plt.close()


    def plot_attempts(self,
                      task_id: str, 
                      grid: bool = False):

        input_true, output_true, attempt_1, attempt_2 = self.get_attempts(task_id)

        if not grid:
            linewidth = 0
        else:
            linewidth = 0.5
        
        fig,ax = plt.subplots(1,4)
        sns.heatmap(input_true, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[0])
        ax[0].set_title('Input', fontsize = 12, fontweight = 'bold')
        sns.heatmap(output_true, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[1])
        ax[1].set_title('Output true', fontsize = 12, fontweight = 'bold')
        sns.heatmap(attempt_1, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[2])
        ax[2].set_title('Attempt 1', fontsize = 12, fontweight = 'bold')
        sns.heatmap(attempt_2, linewidths = linewidth, cmap = self.cmap, norm = self.norm, ax = ax[3])
        ax[3].set_title('Attempt 2', fontsize = 12, fontweight = 'bold')
        fig.set_size_inches(15,3)

        plt.suptitle(f'Task {task_id}\n', 
                     fontsize = 16, 
                     fontweight = 'bold', 
                     y = 1.05, # padding between title and plot
                     color = 'black')
        plt.show()
        plt.close()


    def plot_attempts_all(self,
                          task_id_list,
                          grid: bool = False):
    
        for task_id in task_id_list:
            in_true, solution, out_pred_1, out_pred_2 = self.get_attempts(task_id)
        
            '''
            print("\n--- True ---")
            print(in_true)
            print("\n--- Attempt 1 ---")
            print(out_pred_1)
            print("\n--- Attempt 2 ---")
            print(out_pred_2)
            '''
        
            print(f"Task {task_id}")
            self.plot_attempts(task_id, grid = grid)
            print("\n------------------------------------------------------------------\n")


    def plot_demonstrations(self,
                            task_id: str,
                            grid: bool = False):

        '''
        Plot all demonstrations for a given task.
        '''
    
        input_test, solution, inputs, outputs = self.get_test_pair_with_demos(task_id)

        if not grid:
            linewidth = 0
        else:
            linewidth = 0.5
    
        fig,ax = plt.subplots(1,len(inputs))
        fig.set_size_inches(len(inputs)*3,2)
        for i in range(len(inputs)):
            sns.heatmap(inputs[i], 
                        linewidths = linewidth, 
                        cmap = self.cmap, 
                        norm = self.norm, 
                        ax = ax[i])
            ax[i].set_title('Input', fontsize = 12, fontweight = 'bold')
        plt.suptitle(f'Demonstrations: Task {task_id}\n', 
                     fontsize = 16, 
                     fontweight = 'bold', 
                     y=1.15, # padding between title and plot
                     color = 'black')
        plt.show()
        plt.close()
    
        fig,ax = plt.subplots(1,len(inputs))
        fig.set_size_inches(len(inputs)*3,2)
        for i in range(len(outputs)):
            sns.heatmap(outputs[i], 
                        linewidths = linewidth, 
                        cmap = self.cmap, 
                        norm = self.norm, 
                        ax = ax[i])
            ax[i].set_title('Output', fontsize = 12, fontweight = 'bold')
        plt.show()
        plt.close()


    def rand_str(self, 
                 length: int = 6) -> str:
        return ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(length))
