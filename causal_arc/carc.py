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


class CausalARC:
    

    def __init__(self):

        self.u = UtilsARC()
        self.a = AugmentARC()
        

    def get_ttt_data(self,
                     demo_inputs: list, 
                     demo_outputs: list, 
                     cf_dict: dict) -> dict:

        '''
        Get TTT training data, formatted such that each original input-output
        pair is assigned a single random counterfactual type. Additionally,
        we enforce that all counterfactual types are represented before repeating.
        '''
        
        train_list = []
    
        # Permute counterfactual types.
        cf_types = list(cf_dict.keys())
        idxs = np.random.permutation(np.arange(len(cf_types))).tolist()
        cf_types = [cf_types[i] for i in idxs]
        factor = len(demo_inputs) / len(cf_types)
        if factor > 1:
            cf_types = cf_types * math.ceil(factor)
    
        # Assign random counterfactual type per example.
        for i in range(len(demo_inputs)):
            grid_dict = {"input": demo_inputs[i].tolist(), 
                         "output": demo_outputs[i].tolist()}
            grid_dict["cf_type"] = cf_types[i]
            grid_dict["cf"] = {"input": cf_dict.get(cf_types[i]).get("input")[i].tolist(), 
                               "output": cf_dict.get(cf_types[i]).get("output")[i].tolist()}
            train_list.append(grid_dict)
    
        return train_list


    def get_ttt_data_all_cf(self,
                            demo_inputs: list, 
                            demo_outputs: list, 
                            input_test: np.array,
                            cf_dict: dict) -> dict:

        '''
        Get TTT training data, formatted such that each original input-output
        pair is assigned a list of counterfactuals, represeting every counterfactual
        type for that task.
        '''
        
        train_list = []
        for i in range(len(demo_inputs)):
            grid_dict = {"input": demo_inputs[i].tolist(), 
                         "output": demo_outputs[i].tolist()}
            grid_dict["cf_types"] = list(cf_dict.keys())
            cf_inputs = []
            cf_outputs = []
            cf_types = []
            for key,val in cf_dict.items():
                cf_types.append(key)
                cf_inputs.append(val["input"][i].astype(int).tolist())
                cf_outputs.append(val["output"][i].astype(int).tolist())
            grid_dict["cf_types"] = cf_types
            grid_dict["cf_inputs"] = cf_inputs
            grid_dict["cf_outputs"] = cf_outputs
            train_list.append(grid_dict)

        data_dict = {"test": [{"input": input_test.tolist()}],
                     "train": train_list}
    
        return data_dict
        

    # -*- Prompt generation -*-


    def get_header(self, cf_type: str) -> str:

        '''
        Helper function for prompt generation.
        '''
        
        if "do_hard" in cf_type:
            header = "Counterfactual: Now imagine that we intervened on the original input by fixing some values.\n"
        elif "transform" in cf_type:
            header = "Counterfactual: Now imagine that we intervened on the original input by rotating or flipping it.\n"
        elif "color" in cf_type:
            header = "Counterfactual: Now imagine that we intervened on the original input by changing some colors.\n"
        elif "fun" in cf_type:
            header = "Counterfactual: Now imagine that we intervened on the original input by changing the causal function for some elements.\n"
        else:
            print("Cannot locate header.")
        return header

    
    def get_prompt_induction(self,
                             sample_dict: dict, 
                             n_examples: int = 1,
                             counterfactuals: bool = False,
                             n_counterfactuals: int = 3,
                             scm: bool = False) -> str:

        intro = "You must solve the following puzzle by discovering the deterministic rule that maps inputs to outputs. "
        intro += "Both the inputs and outputs are 2D Python arrays of colored pixels. "
        if not counterfactuals:
            intro += "We provide example input-output pairs as demonstration. "
        else: 
            intro += "We provide example input-output pairs along with counterfactual examples, "
            intro += "which represent interventions on the original examples. "
        intro += "To solve the problem, express the deterministic rule as a Python program. Do not explain your reasoning, and only output a single Python program.\n"

        prompt_str = [intro]

        # Add a string representation of the SCM, if specified.
        if scm:
            scm_header = "First, we provide the structural causal model as a Python program:\n"
            prompt_str.append(scm_header+sample_dict["scm"]+"\n")

        # Get example pairs.
        example_header = "Example input-output arrays:\n"
        example_idx = np.random.choice(list(range(len(sample_dict["train"]))), size = n_examples)
        train = sample_dict["train"].copy()
        example_pairs = [(train[i]["input"],train[i]["output"]) for i in example_idx]
        for i in range(n_examples):
            example_pair = example_pairs[i]
            example_str = example_header + str(example_pair[0]) + " -> " + str(example_pair[1]) + "\n"
            prompt_str.append(example_str)

            # Add counterfactuals for current example pair.
            if counterfactuals:
                cf_inputs = train[i]["cf_inputs"].copy()
                cf_outputs = train[i]["cf_outputs"].copy()
                cf_types = train[i]["cf_types"].copy()
                max_cfs = len(cf_inputs)
                if n_counterfactuals > max_cfs:
                    warnings.warn(f"n_counterfactuals cannot exceed {max_cfs}, reducing to {max_cfs}")
                    n_counterfactuals = max_cfs
                cf_idx = np.random.choice(list(range(len(cf_inputs))), 
                                          size = n_counterfactuals, 
                                          replace = False)
                for idx in cf_idx:
                    pair = (cf_inputs[idx],cf_outputs[idx])
                    header = self.get_header(cf_types[idx])
                    if n_examples > 1:
                        header = header.replace("original", "previous")
                    cf_str = header + str(pair[0]) + " -> " + str(pair[1]) + "\n"
                    prompt_str.append(cf_str)
            
        prompt_str = "".join(prompt_str)
        return prompt_str
        

    def get_prompt_discovery(self,
                             sample_dict: dict, 
                             n_examples: int = 1,
                             counterfactuals: bool = False,
                             n_counterfactuals: int = 3,
                             scm: bool = False) -> str:

        intro = "You must solve the following causal discovery problem, where the cells in an "
        intro += "input array are causal parents of cells in an output array. "
        intro += "Both the inputs and outputs are 2D Python arrays of colored pixels. "
        if not counterfactuals:
            intro += "We provide example input-output pairs as demonstration. "
        else: 
            intro += "We provide example input-output pairs along with counterfactual examples, "
            intro += "which represent interventions on the original examples. "
        intro += "You must predict the causal function(s) that relate parent cells in the input to their children in the output. "
        intro += "Be concise: do not explain your reasoning, and start your answer with 'The logical operators are'.\n"

        prompt_str = [intro]

        # Add a string representation of the SCM, if specified.
        if scm:
            scm_header = "First, we provide the structural causal model as a Python program:\n"
            prompt_str.append(scm_header+sample_dict["scm"]+"\n")

        # Get example pairs.
        example_header = "Example input-output arrays:\n"
        example_idx = np.random.choice(list(range(len(sample_dict["train"]))), size = n_examples)
        train = sample_dict["train"].copy()
        example_pairs = [(train[i]["input"],train[i]["output"]) for i in example_idx]
        for i in range(n_examples):
            example_pair = example_pairs[i]
            example_str = example_header + str(example_pair[0]) + " -> " + str(example_pair[1]) + "\n"
            prompt_str.append(example_str)

            # Add counterfactuals for current example pair.
            if counterfactuals:
                cf_inputs = train[i]["cf_inputs"].copy()
                cf_outputs = train[i]["cf_outputs"].copy()
                cf_types = train[i]["cf_types"].copy()
                max_cfs = len(cf_inputs)
                if n_counterfactuals > max_cfs:
                    warnings.warn(f"n_counterfactuals cannot exceed {max_cfs}, reducing to {max_cfs}")
                    n_counterfactuals = max_cfs
                cf_idx = np.random.choice(list(range(len(cf_inputs))), 
                                          size = n_counterfactuals, 
                                          replace = False)
                for idx in cf_idx:
                    pair = (cf_inputs[idx],cf_outputs[idx])
                    header = self.get_header(cf_types[idx])
                    cf_str = header + str(pair[0]) + " -> " + str(pair[1]) + "\n"
                    prompt_str.append(cf_str)
            
        prompt_str = "".join(prompt_str)
        return prompt_str
        
    def get_prompt_cf_reasoning(self,
                                sample_dict: dict, 
                                n_examples: int = 1,
                                counterfactuals: bool = False,
                                n_counterfactuals: int = 3,
                                scm: bool = False) -> dict:

        intro = "You must solve the following puzzle by discovering the deterministic rule that maps inputs to outputs. "
        intro += "You will then be asked to predict the output for a counterfactual example. "
        intro += "Both the inputs and outputs are 2D grids of colored pixels. "
        if not counterfactuals:
            intro += "We provide example input-output pairs as demonstration. "
        else:
            intro += "We provide example input-output pairs along with counterfactual examples, "
            intro += "which represent interventions on the original examples. "
        intro += "Grids are provided as Python arrays. You must output only a single Python array, "
        intro += "and do not explain your reasoning.\n"

        prompt_str = [intro]

        # Add a string representation of the SCM, if specified.
        if scm:
            scm_header = "First, we provide the structural causal model as a Python program:\n"
            prompt_str.append(scm_header+sample_dict["scm"]+"\n")

        # Get example pairs.
        example_header = "Example input-output arrays:\n"
        example_idx = np.random.choice(list(range(len(sample_dict["train"]))), size = n_examples)
        train = sample_dict["train"].copy()
        for idx in example_idx:
            example_pair = (train[idx]["input"],train[idx]["output"])
            example_str = example_header + str(example_pair[0]) + " -> " + str(example_pair[1]) + "\n"
            prompt_str.append(example_str)

            # Cf metadata.
            cf_inputs = train[idx]["cf_inputs"].copy()
            cf_outputs = train[idx]["cf_outputs"].copy()
            cf_types = train[idx]["cf_types"].copy()

            # Add counterfactuals for current example pair, unless it is the final pair.
            if counterfactuals: #and idx != example_idx[-1]:
                max_cfs = len(cf_inputs)-1
                if n_counterfactuals > max_cfs:
                    warnings.warn(f"n_counterfactuals cannot exceed {max_cfs}, reducing to {max_cfs}")
                    n_counterfactuals = max_cfs

                cf_idx = np.random.choice(list(range(len(cf_inputs))), 
                                          size = n_counterfactuals+1, 
                                          replace = False)
                
                # Hold out one cf for test pair.
                held_out_idx = cf_idx[0]
                held_out_cf_input = cf_inputs[held_out_idx]
                held_out_cf_output = cf_outputs[held_out_idx]
                held_out_cf_type = cf_types[held_out_idx]

                # Add counterfactual demonstrations.
                cf_idx = cf_idx[1:]
                for idx in cf_idx:
                    pair = (cf_inputs[idx],cf_outputs[idx])
                    header = self.get_header(cf_types[idx])
                    if n_examples > 1:
                        header = header.replace("original", "previous")
                    cf_str = header + str(pair[0]) + " -> " + str(pair[1]) + "\n"
                    prompt_str.append(cf_str)
            else:
                cf_idx = np.random.choice(list(range(len(cf_inputs))), size = 1, replace = False)[0]
                held_out_cf_input = cf_inputs[cf_idx]
                held_out_cf_output = cf_outputs[cf_idx]
                held_out_cf_type = cf_types[cf_idx]

        # Get string rep of test pair (this time, the held out cf).
        header = self.get_header(held_out_cf_type)
        header = header.replace("original", "previous")
        test_str = header + str(held_out_cf_input) + " -> "
        prompt_str.append(test_str)
            
        prompt_str = "".join(prompt_str)
        return {"prompt": prompt_str, 
                "test": {"input": held_out_cf_input,
                         "output": held_out_cf_output}}

    
    def get_prompt_abstract_reasoning(self,
                                      sample_dict: dict, 
                                      n_examples: int = 1,
                                      counterfactuals: bool = False,
                                      n_counterfactuals: int = 3,
                                      scm: bool = False) -> str:

        intro = "You must solve the following puzzle by discovering the deterministic rule that maps inputs to outputs. "
        intro += "Both the inputs and outputs are 2D grids of colored pixels. "
        if not counterfactuals:
            intro += "We provide example input-output pairs as demonstration. "
        else:
            intro += "We provide example input-output pairs along with counterfactual examples, "
            intro += "which represent interventions on the original examples. "
        intro += "Grids are provided as Python arrays. You must output only a single Python array, "
        intro += "and do not explain your reasoning.\n"

        prompt_str = [intro]

        # Add a string representation of the SCM, if specified.
        if scm:
            scm_header = "First, we provide the structural causal model as a Python program:\n"
            prompt_str.append(scm_header+sample_dict["scm"]+"\n")

        # Get example pairs.
        example_header = "Example input-output arrays:\n"
        example_idx = np.random.choice(list(range(len(sample_dict["train"]))), size = n_examples)
        train = sample_dict["train"].copy()
        example_pairs = [(train[i]["input"],train[i]["output"]) for i in example_idx]
        for i in range(n_examples):
            example_pair = example_pairs[i]
            example_str = example_header + str(example_pair[0]) + " -> " + str(example_pair[1]) + "\n"
            prompt_str.append(example_str)

            # Add counterfactuals for current example pair.
            if counterfactuals:
                cf_inputs = train[i]["cf_inputs"].copy()
                cf_outputs = train[i]["cf_outputs"].copy()
                cf_types = train[i]["cf_types"].copy()
                max_cfs = len(cf_inputs)
                if n_counterfactuals > max_cfs:
                    warnings.warn(f"n_counterfactuals cannot exceed {max_cfs}, reducing to {max_cfs}")
                    n_counterfactuals = max_cfs
                cf_idx = np.random.choice(list(range(len(cf_inputs))), 
                                          size = n_counterfactuals, 
                                          replace = False)
                for idx in cf_idx:
                    pair = (cf_inputs[idx],cf_outputs[idx])
                    header = self.get_header(cf_types[idx])
                    cf_str = header + str(pair[0]) + " -> " + str(pair[1]) + "\n"
                    prompt_str.append(cf_str)

        # Get string rep of test pair.
        test_pair = (sample_dict["test"][0]["input"], sample_dict["test"][0]["output"])
        test_header = "Test input-output arrays:\n"
        test_str = test_header + str(test_pair[0]) + " -> "
        prompt_str.append(test_str)
            
        prompt_str = "".join(prompt_str)
        return prompt_str
    
    
    def get_prompt(self,
                   sample_dict: dict, 
                   n_examples: int = 1,
                   counterfactuals: bool = False,
                   n_counterfactuals: int = 3,
                   scm: bool = False,
                   problem_type: str = "cf_reasoning") -> str:

        # Intro based on task.
        if problem_type == "cf_reasoning":
            prompt = self.get_prompt_cf_reasoning(sample_dict = sample_dict, 
                                                  n_examples = n_examples,
                                                  counterfactuals = counterfactuals,
                                                  n_counterfactuals = n_counterfactuals,
                                                  scm = scm)
            return prompt
        elif problem_type == "abstract_reasoning":
            prompt = self.get_prompt_abstract_reasoning(sample_dict = sample_dict, 
                                                        n_examples = n_examples,
                                                        counterfactuals = counterfactuals,
                                                        n_counterfactuals = n_counterfactuals,
                                                        scm = scm)
            return prompt
        elif problem_type == "discovery":
            prompt = self.get_prompt_discovery(sample_dict = sample_dict, 
                                               n_examples = n_examples,
                                               counterfactuals = counterfactuals,
                                               n_counterfactuals = n_counterfactuals,
                                               scm = scm)
            return prompt
        elif problem_type == "induction":
            prompt = self.get_prompt_induction(sample_dict = sample_dict, 
                                               n_examples = n_examples,
                                               counterfactuals = counterfactuals,
                                               n_counterfactuals = n_counterfactuals,
                                               scm = scm)
            return prompt
            
        else:
            raise ValueError("problem_type must be in ['counterfactual_reasoning,abstract_reasoning,discovery,induction']")
        

    

    
    

    

    