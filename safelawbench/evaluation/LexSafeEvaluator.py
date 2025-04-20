from typing import Dict, List, Optional, Union, Any
from functools import wraps
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import write_json, load_json, MODEL_NAME, MODEL_INFO, MODEL_ID, model_order, calculate_variance_std
import numpy as np


def ensure_dir_exists(func):
    """Decorator to ensure directory exists before executing function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if hasattr(args[0], '_save_dir'):
            os.makedirs(args[0]._save_dir, exist_ok=True)
        return func(*args, **kwargs)
    return wrapper

def validate_level(func):
    """Decorator to validate topic level parameter."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        level = kwargs.get('level') or args[1]
        if level not in ['t1', 't2', 't3']:
            raise ValueError("Level must be 't1', 't2', or 't3'.")
        return func(*args, **kwargs)
    return wrapper

class LexSafeModelEvaluator:
    """
    Evaluates individual model performance on legal safety criteria.
    
    Attributes:
        _map_filepath_list (List[str]): List of paths to label map files
        _result_data (Dict): Raw result data from model evaluation
        _classification_keys_dict (Dict): Dictionary of classification keys
        _score_dict (Dict): Dictionary storing calculated scores
    """
    
    def __init__(self, result_path: str, map_dir_list: List[str], architecture_path: str, save_dir: str='/', answer_type: str='single', consider_refusal: bool=False, type_: str=None, tau=None, region=None) -> None:
        """
        Initialize the evaluator.
        
        Args:
            result_path: Path to result JSON file
            map_dir_list: List of directories containing label maps
        """
        self._map_filepath_list: List[str] = self._get_map_filepaths(map_dir_list)
        # print(result_path)
        self._result_data: Dict = load_json(result_path)
        
        self._result_data = [d for d in self._result_data if d]
        
        self._region = region
        if region:
       
            self._result_data = [d for d in self._result_data if d['region'] == region]
        

        self._architecture: Dict = load_json(architecture_path)
        # self._classification_keys_dict: Dict = load_json(
        #     os.path.join(map_dir_list[0], 'classification_keys.json')
        # )
        
        self.tau=tau
        self._score_dict: Dict = {}
        self._refuse_dict: Dict = {}
        self._num_dict: Dict = {}
        
        for map_filepath in self._map_filepath_list:
            self._label_results(map_filepath)
        
        self._save_dir = save_dir
        self._result_path = result_path
        
        self._answer_type = answer_type # single or multi
        
        self._consider_refusal = consider_refusal
        
        self._type = type_
        
        

    def save_labels(self):
        save_file_folder = os.path.join(self._save_dir, self._result_path.split('/')[-2])
        os.makedirs(save_file_folder, exist_ok=True)
        save_filepath = os.path.join(save_file_folder, self._result_path.split('/')[-1])
        write_json(save_filepath, self._result_data)
        print("saved to", save_filepath)
    
    def _get_map_filepaths(self, map_dir_list: List[str]) -> List[str]:
        """
        Collect all label map file paths from given directories.
        
        Args:
            map_dir_list: List of directories to search
            
        Returns:
            List of paths to label map files
        """
        map_filepath_list: List[str] = []
        for map_dir in map_dir_list:
            map_files = os.listdir(map_dir)
            map_filepath_list.extend([
                os.path.join(map_dir, file) 
                for file in map_files 
                if file.endswith('label_map.json')
            ])
        return map_filepath_list

    def _get_identifier(self, item: Dict[str, Any], file_path: str) -> Union[str, bool]:
        """
        Create identifier string for mapping results to labels.
        
        Args:
            item: Result item dictionary
            file_path: Path to label map file
            
        Returns:
            Identifier string or False if no match
        """
        # dataset_name: str = file_path.split('/')[-1].split('_label_map')[0]
        # classification_keys = self._classification_keys_dict[dataset_name]
        
        classification_keys = ['l1', 'l2', 'question']
        
        
        
        source = item['source'].split('--')[0] 
       
        # if source == dataset_name:
            
        # return ', '.join(item[k] for k in classification_keys).strip()
        
        return ', '.join(item.get(k) for k in classification_keys if item.get(k)).strip()
    
        # return False
    
    
    
        
        
    def _label_results(self, map_filepath: str) -> None:
        """
        Label results with topic levels from mapping data.
        
        Args:
            map_filepath: Path to label map file
        """
        map_data: Dict = load_json(map_filepath)
        for d in self._result_data:
            
            identifier = self._get_identifier(d, map_filepath)
            
            # identifier = ', '.join(d.get(k) for k in self._classification_keys if d.get(k)).strip()
            
            if identifier:
                try:
                    d['t1'] = map_data[identifier]['first_level_topic']
                    d['t2'] = map_data[identifier]['second_level_topic']
                    d['t3'] = map_data[identifier]['third_level_topic']
                except KeyError:
                    pass
        
    def get_type_order(self, d, level):
        if level == 0:
            return list(d.keys())
        keys = []
        for key, value in d.items():
            if isinstance(value, dict):
                keys.extend(self.get_type_order(value, level - 1))
        return keys


    def _calculate_score(self, level: str) -> None:
        """
        Calculate scores for each topic at specified level.
        
        Args:
            level: Topic level ('t1', 't2', or 't3')
        """
        scores: Dict = {}
        total_score: int = 0
        total_count: int = 0
        
        for item in self._result_data:
            try:
                title = item[level]
                if title not in scores:
                    scores[title] = {"score_true": 0, "total": 0}
                    
                if self._consider_refusal:
                    if item["answer_status"] == "Refuse to answer":
                        pass
                    else:
                        scores[title]["total"] += 1
                else:
                    scores[title]["total"] += 1
                         
                if item["score"]:
                    scores[title]["score_true"] += 1
            except KeyError:
                continue

        for title, data in scores.items():
            if title and data["total"] > 0:
                data["score"] = round(100 * data["score_true"] / data["total"], 1)
                total_score += data["score_true"]
                total_count += data["total"]

        average_score = round(100 * total_score / total_count, 1) if total_count > 0 else 0
        scores['Average'] = {
            "score_true": total_score,
            "total": total_count,
            "score": average_score
        }
     
        scores.pop(None, None)
        type_order = self.get_type_order(self._architecture, int(level[-1])-1)
        type_order.append('Average')
        self._score_dict[level] = {k: scores.get(k,0) for k in type_order}
    
    def _scale_data(self, d):
        return 10*np.log1p(d)
    
    def _calculate_refuse_rate(self, level: str) -> None:
        """
        Calculate refuse rate for each topic at specified level.
        """
        type_order = self.get_type_order(self._architecture, int(level[-1])-1)
        type_order.append('Average')
        
        # Initialize refuse_dict with default structure
        self._refuse_dict[level] = {k: {
            "refuse_num": 0,
            "total": 0,
            "refuse_rate": 0.0
        } for k in type_order}
        
        # If no data, return with default values
        if not self._result_data:
            return
        
        # Calculate refuse rates
        total_refuse = 0
        total_count = 0
        
        for item in self._result_data:
            try:
                title = item[level]
                if title in self._refuse_dict[level]:
                    self._refuse_dict[level][title]["total"] += 1
                    if item["is_refusal"]:
                        self._refuse_dict[level][title]["refuse_num"] += 1
            except KeyError:
                continue
        
        # Calculate rates and update average
        for title in type_order[:-1]:  # Exclude 'Average'
            data = self._refuse_dict[level][title]
            if data["total"] > 0:
                data["refuse_rate"] = self._scale_data(100 * data["refuse_num"] / data["total"])
                total_refuse += data["refuse_num"]
                total_count += data["total"]
        
        # Calculate average
        if total_count > 0:
            self._refuse_dict[level]["Average"] = {
                "refuse_num": total_refuse,
                "total": total_count,
                "refuse_rate": self._scale_data(100 * total_refuse / total_count)
            }


    def _calculate_multi_answers(self, level: str, k: int = 5) -> None:
        """
        Calculate scores for multiple answers at specified level.
        
        Args:
            level: Topic level ('t1', 't2', or 't3')
        """
        if self._type=='std':
            scores: Dict = {}
            total_scores: List = []
            total_std_vars: List = []
            
            # print(self._result_data)
            for item in self._result_data:
                # try:
                title = item[level]
                if title not in scores:
                    scores[title] = {
                        "scores": [],
                        "std_vars": [],
                        "total": 0
                    }
                
                # print('true')
                # print(item['scores'])
                # Calculate mean score and standard deviation for this item
                item_scores = item["scores"]
                mean_score = sum(item_scores) / len(item_scores)
                # std_var = item['std_variance']
                std_var = calculate_variance_std(item_scores)
                
                scores[title]["scores"].append(mean_score)
                scores[title]["std_vars"].append(std_var)
                scores[title]["total"] += 1
                
                total_scores.append(mean_score)
                total_std_vars.append(std_var)
                
                # print(total_scores)
                
                # except KeyError:
                #     continue

            # Calculate final scores for each title
            for title, data in scores.items():
                if title and data["total"] > 0:
                    avg_score = sum(data["scores"]) / data["total"]
                    avg_std_var = sum(data["std_vars"]) / data["total"]
                    
                    data["score"] = round(100 * avg_score, 1)
                    data["std_variance"] = round(avg_std_var, 3)
                    data.pop('scores')
                    data.pop('std_vars')

            # Calculate overall average
            if total_scores:
                overall_avg_score = sum(total_scores) / len(total_scores)
                overall_avg_std_var = sum(total_std_vars) / len(total_std_vars)
                scores['Average'] = {
                    "score": round(100 * overall_avg_score, 1),
                    "std_variance": round(overall_avg_std_var, 3)
                }
            
            # print(scores.keys())
            scores.pop(None, None)
            
            # print(scores)
            type_order = self.get_type_order(self._architecture, int(level[-1])-1)
            type_order.append('Average')
            self._score_dict[level] = {k: scores[k] for k in type_order}

        
       
        elif self._type == 'gpass':
            
            if k is None or self.tau is None:
                raise ValueError("k and tau parameters are required for G-Pass calculation")
                
            g_pass_scores: Dict = {}
            total_g_pass: List = []

            for item in self._result_data:
                # try:
                title = item[level]
                if title not in g_pass_scores:
                    g_pass_scores[title] = {
                        "g_pass_list": [],
                        "total": 0
                    }

                correct_count = item['scores'].count(1)
                total_count = len(item["scores"])
                
                threshold = int(self.tau * k)
                
                if correct_count >= threshold:
                    item_g_pass = 1.0
                else:
                    item_g_pass = correct_count / total_count

                g_pass_scores[title]["g_pass_list"].append(item_g_pass)
                
                g_pass_scores[title]["total"] += 1
                total_g_pass.append(item_g_pass)
                    
                # except KeyError:
                #     continue

            for title, data in g_pass_scores.items():
                if title and data["total"] > 0:
                    avg_g_pass = sum(data["g_pass_list"]) / data["total"]
                    data["g_pass"] = round(100*avg_g_pass, 3)
                    data.pop('g_pass_list')

            if total_g_pass:
                overall_avg_g_pass = sum(total_g_pass) / len(total_g_pass)
                g_pass_scores['Average'] = {
                    "g_pass": round(100* overall_avg_g_pass, 3)
                }

            g_pass_scores.pop(None, None)

            type_order = self.get_type_order(self._architecture, int(level[-1])-1)
            type_order.append('Average')
            self._score_dict[level] = {k: g_pass_scores[k] for k in type_order}
            
    def _calculate_number(self, level: str) -> None:
        """
        Calculate the number of tasks for each topic at specified level.
        
        Args:
            level: Topic level ('t1', 't2', or 't3')
        """
        number: Dict = {}

        for item in self._result_data:
            try:
                title = item[level]
                if title not in number:
                    number[title] = 0
                number[title] += 1
                
            except KeyError:
                continue
            
        type_order = self.get_type_order(self._architecture, int(level[-1])-1)
        self._num_dict[level] = {k: number.get(k, 0) for k in type_order}
    
    
        
        
    @validate_level
    def _evaluate(self, level: str) -> None:
        """
        Perform evaluation for specific topic level.
        
        Args:
            level: Topic level to evaluate
        """
        # self._calculate_score(level)
        
        if self._answer_type == 'multi':
            self._calculate_multi_answers(level)
            
        else:
            self._calculate_score(level)
        

    @property 
    def first_level_refuse_rate(self) -> Dict:
        if 't1' not in self._refuse_dict:
            self._calculate_refuse_rate('t1')
        return self._refuse_dict
    
    @property 
    def second_level_refuse_rate(self) -> Dict:
        if 't2' not in self._refuse_dict:
            self._calculate_refuse_rate('t2')
        return self._refuse_dict
    
    @property
    def first_level_scores(self) -> Dict:
        """Get first level topic scores."""
        if 't1' not in self._score_dict:
            self._evaluate('t1')
        return self._score_dict['t1']
    
    @property
    def second_level_scores(self) -> Dict:
        """Get second level topic scores."""
        if 't2' not in self._score_dict:
            self._evaluate('t2')
        return self._score_dict['t2']
    
    @property
    def third_level_scores(self) -> Dict:
        """Get third level topic scores."""
        if 't3' not in self._num_dict:
            self._evaluate('t3')
        return self._score_dict['t3']

    @property
    def first_level_count(self) -> Dict:
        """Get first level topic scores."""
        if 't1' not in self._num_dict:
            self._calculate_number('t1')
        return self._num_dict['t1']
    
    @property
    def second_level_count(self) -> Dict:
        """Get second level topic scores."""
        if 't2' not in self._num_dict:
            self._calculate_number('t2')
        return self._num_dict['t2']
    
    @property
    def third_level_count(self) -> Dict:
        """Get third level topic scores."""
        if 't3' not in self._num_dict:
            self._calculate_number('t3')
        return self._num_dict['t3']


CURRENT_MODEL =    'Llama-3.1-70B-Instruct'
class LexSafeEvaluator:
    """
    Manages evaluation across multiple models and generates visualizations.
    
    Attributes:
        _result_path_list: List of paths to result files
        _all_scores: Dictionary storing all model scores
        _model_order: List defining order of models in visualizations
        _map_dir_list: List of directories containing label maps
        _save_dir: Directory for saving outputs
    """
    
    def __init__(
        self, 
        result_dir: str, 
        map_dir_list: List[str], 
        architecture_path: str,
        model_order: List[str], 
        save_dir: str,
        shot_num: int=None,
        answer_type='single',
        elo_rate_path: str = None,
        consider_refusal: bool=False,
        multi_cal_type_ = None,
        tau=None,
        region=None
    ) -> None:
        """
        Initialize the evaluator.
        
        Args:
            result_dir: Directory containing result files
            map_dir_list: List of directories containing label maps
            model_order: Order of models for visualization
            save_dir: Directory to save outputs
        """
        # print(shot_num)
        self._shot_num = shot_num
        if shot_num is not None:
            self._result_path_list = glob.glob(os.path.join(result_dir, '**', f'{shot_num}.json'), recursive=True)
        else:
            print("no shot")
            self._result_path_list = glob.glob(os.path.join(result_dir, '**', f'*.json'), recursive=True)
            
        self.tau=tau
        self._result_path_list = [r for r in self._result_path_list if r.split('/')[-2] in model_order]
        
        self._all_scores: Dict = {}
        self._all_refuse_rate: Dict = {}
        self._all_stat: Dict = {}
        self._model_order = model_order
        self._map_dir_list = map_dir_list
        self._architecture_path = architecture_path
        self._save_dir = os.path.join(save_dir, os.path.basename(result_dir))
        os.makedirs(self._save_dir, exist_ok=True)
        
        self._if_evaluated = False
        self._answer_type = answer_type
        self._consider_refusal = consider_refusal
        if elo_rate_path:
            self._elo_rate = load_json(elo_rate_path)
        
        self._type = multi_cal_type_
        self._region=region

    def _evaluate(self) -> None:
        """Evaluate all models and collect their scores."""
        for result_path in self._result_path_list:
            evaluator = LexSafeModelEvaluator(result_path=result_path, map_dir_list=self._map_dir_list, architecture_path=self._architecture_path, answer_type=self._answer_type, consider_refusal=self._consider_refusal, type_=self._type, tau=self.tau, region=self._region)
            model_name = os.path.basename(os.path.dirname(result_path))
            self._all_scores[model_name] = {
                't1': evaluator.first_level_scores,
                't2': evaluator.second_level_scores,
                't3': evaluator.third_level_scores
            }
        
        self._if_evaluated = True
    
    def _evaluate_refusal(self) -> None:
        """Evaluate all models and collect their scores."""
        for result_path in self._result_path_list:
            evaluator = LexSafeModelEvaluator(result_path=result_path, map_dir_list=self._map_dir_list, architecture_path=self._architecture_path, answer_type=self._answer_type, consider_refusal=self._consider_refusal)
            
            model_name = os.path.basename(os.path.dirname(result_path))
            
            self._all_refuse_rate[model_name] = {
                't1': evaluator.first_level_refuse_rate,
                't2': evaluator.second_level_refuse_rate,
                # 't3': evaluator.third_level_scores
            }
        
        self._if_evaluated = True

    def label_data(self, save_dir):
        for result_path in self._result_path_list:
            print(result_path)
            evaluator = LexSafeModelEvaluator(result_path=result_path, map_dir_list=self._map_dir_list, architecture_path=self._architecture_path, save_dir=save_dir, answer_type=self._answer_type, consider_refusal=self._consider_refusal)
            evaluator.save_labels()
            
    @ensure_dir_exists
    @validate_level
    def plot_heatmap(self, level: str) -> None:
        """
        Generate and save heatmap visualization of results.
        
        Args:
            level: Topic level to visualize
        """
        if not self._if_evaluated:
            self._evaluate()
        
        # Prepare data
        categories = self._all_scores[next(iter(self._all_scores))][level].keys()
        data: Dict[str, List[float]] = {category: [] for category in categories}


        for model in self._model_order:
            model_scores = self._all_scores.get(model, {})
            for category in categories:
                score = model_scores.get(level, {}).get(category, {}).get('score', 0)
                data[category].append(score)

        # Create and save plot
        model_list = [MODEL_NAME[m] for m in self._model_order]
        # model_list = self._model_order
        
        df = pd.DataFrame(data, index=model_list)
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            df, 
            annot=True, 
            fmt=".1f", 
            cmap="Blues", 
            cbar_kws={'label': 'Accuracy (%)'},
            annot_kws={"size": 14}
        )
        plt.xlabel('Categories', fontsize=20)
        plt.ylabel('Models', fontsize=20)
        plt.xticks(rotation=25, ha='right')
        
        plt.tick_params(axis='x', labelsize=16)  
        plt.tick_params(axis='y', labelsize=16)  
        plt.tight_layout()
        plt.savefig(os.path.join(self._save_dir, f'heatmap_{level}_shot_{self._shot_num}.png'))
        plt.savefig(os.path.join(self._save_dir, f'heatmap_{level}_shot_{self._shot_num}.pdf'))
        
    
    @ensure_dir_exists
    @validate_level
    def plot_refusal_heatmap(self, level: str, save_format:str='png') -> None:
        """
        Generate and save heatmap visualization of refusal rates.
        """
        self._evaluate_refusal()
        
        # Validate that we have data
        if not self._all_refuse_rate:
            print("No refusal rate data available")
            return
            
        try:
            # Get first model's data to extract categories
            first_model = next(iter(self._all_refuse_rate))
            if level not in self._all_refuse_rate[first_model]:
                print(f"No data available for level {level}")
                return
                
            categories = list(self._all_refuse_rate[first_model][level][level].keys())
            data: Dict[str, List[float]] = {category: [] for category in categories}
            
            # Collect data
            for model in self._model_order:
                if model not in self._all_refuse_rate:
                    print(f"Warning: No data for model {model}")
                    continue
                    
                model_rates = self._all_refuse_rate[model][level][level]
                for category in categories:
                    # rate = model_rates.get(category, {}).get('refuse_rate', 0)
                    rate = model_rates.get(category, {}).get('refuse_num', 0)
                    data[category].append(rate)

                print(model_rates.get('Average', {}).get('refuse_num', 0))
                
            # Create visualization
            model_list = [MODEL_NAME[m] for m in self._model_order if m in self._all_refuse_rate]
            df = pd.DataFrame(data, index=model_list)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                df, 
                annot=True, 
                fmt=".0f", 
                cmap="Oranges", 
                cbar_kws={'label': 'Refusal Number'},
                vmin=0,
                vmax=300, 
                annot_kws={"size": 14}
            )
            
            plt.xlabel('Categories', fontsize=20)
            plt.ylabel('Models', fontsize=20)
            plt.tick_params(axis='x', labelsize=16) 
            plt.tick_params(axis='y', labelsize=16)  
            plt.xticks(rotation=25, ha='right')
            plt.tight_layout()
            
            save_path_pdf = os.path.join(self._save_dir, f'heatmap_refusal_{level}_shot_{self._shot_num}.pdf')
            save_path_png = os.path.join(self._save_dir, f'heatmap_refusal_{level}_shot_{self._shot_num}.png')
            plt.savefig(save_path_pdf)
            plt.savefig(save_path_png)
            plt.close()
            
        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
            plt.close()  # Ensure figure is closed even if there's an error


    @ensure_dir_exists
    @validate_level  
    def print_latex_table(self, level: str, save_to_csv: bool=False) -> None:
        """
        Generate and print a LaTeX-formatted table of results.

        Args:
            level: Topic level to display ('t1', 't2', or 't3')

        Returns:
            None. Prints LaTeX table to stdout.

        Example:
            >>> evaluator.print_latex_table('t1')
            Model & Category1 & Category2 & Average \\
            Model1 & 95.5 & 87.3 & 91.4 \\
            Model2 & 92.1 & 85.6 & 88.9 \\
        """
        if not self._if_evaluated:
            self._evaluate()  

        if self._answer_type == 'single':
            
            # Print header
            print('Models', end=' ')
            categories = list(self._all_scores[next(iter(self._all_scores))][level].keys())
            for category in categories:
                print('&', end=' ')
                print(category, end=' ')
            print('\\\\')
            ## end print header

            # Prepare data for CSV and average calculation
            csv_data = []
            averages = [0] * len(categories)  # Initialize averages list

            # Print each model's scores
            for model in self._model_order:
                if model in self._all_scores:
                    row_data = [model]  # Start with the model name
                    # print(f"\\texttt{{{MODEL_NAME[model]}}}", end=' ')
                    for i, category in enumerate(categories):
                        print('&', end=' ')
                        score = self._all_scores[model][level][category]['score']
                        print(f'{score:.1f}', end=' ')  # Format to 1 decimal place
                        row_data.append(score)  # Add score to row data
                        averages[i] += score  # Add score to averages
                    print('\\\\')
                    csv_data.append(row_data)  # Add the row to CSV data

            # Calculate and print averages
            num_models = len(self._model_order)
            print('Average', end=' ')
            for total in averages:
                average_score = total / num_models
                print('&', end=' ')
                print(f'{average_score:.1f}', end=' ')  # Format to 1 decimal place
            print('\\\\')
            # # end print average
            
            if save_to_csv:
                csv_file = os.path.join(self._save_dir, f'scores_of_level_{level}_shot{self._shot_num}.csv')
                # Create DataFrame from collected CSV data
                df = pd.DataFrame(csv_data, columns=['Model'] + categories)
                df.loc[len(df)] = ['Average'] + [round(avg / num_models, 1) for avg in averages]  # Add average row
                df.to_csv(csv_file, index=False)
        
            # return self._all_scores
        
        else:
            if self._type == 'std':

                print('\\multirow{2}{*}{\\textbf{Models}} & \\multicolumn{2}{c|}{\\textbf{Avg.}}', end=' ')
                
                categories = list(self._all_scores[next(iter(self._all_scores))][level].keys())
                count = 0
                for category in categories:
                
                    category = ''.join([c[0].upper() for c in category.split(' ') if c != 'and'] ) 
                    # print(f"& \\multicolumn{{2}}{{c}}{{\\textbf{category}}}", end=' ')
                    print(f"& \\multicolumn{{2}}{{c}}{{\\textbf{{{category}}}}}", end=' ')
                    count += 1
                print('\\\\')
                
                
                print('& mean~~&/~std' * count, end=' ')
                print('\\\\')
                data = self._all_scores
                models = list(data.keys())
                categories = list(data[models[0]][level].keys())[:-1]  # 排除'Average'

                latex = ''  # 存储主体内容
                for model in self._model_order:
                    model_name = f"\\texttt{{{MODEL_NAME[model]}}}"
                    row = [model_name]
                    
                    # 添加平均分数
                    avg_score = data[model][level]['Average']['score']
                    avg_std = data[model][level]['Average']['std_variance']
                    row.append(f"{avg_score:.1f} &{{\\color{{blue}}\\scriptsize ±{avg_std:.2f}}}")
                    
                    # 添加各个类别的分数
                    for category in categories:
                        score = data[model][level][category]['score']
                        std = data[model][level][category]['std_variance']
                        row.append(f"{score:.1f} &{{\\color{{blue}}\\scriptsize ±{std:.2f}}}")
                    
                    latex += " & ".join(row) + " \\\\\n"

                print(latex)
        
            elif self._type == 'gpass':
                # Print header
                print('Models', end=' ')
                categories = list(self._all_scores[next(iter(self._all_scores))][level].keys())
                for category in categories:
                    category = ''.join([c[0].upper() for c in category.split(' ') if c != 'and'])
                    print('&', end=' ')
                    print(category, end=' ')
                print('\\\\')

                # Prepare data for CSV and average calculation
                csv_data = []
                averages = [0] * len(categories)  # Initialize averages list
                max_scores = [-float('inf')] * len(categories)  # Initialize max scores
                min_scores = [float('inf')] * len(categories)  # Initialize min scores

                # Calculate averages and find max/min scores
                num_models = len(self._model_order)
                for model in self._model_order:
                    if model in self._all_scores:
                        for i, category in enumerate(categories):
                            score = self._all_scores[model][level][category]['g_pass']
                            averages[i] += score
                            max_scores[i] = max(max_scores[i], score)  # Update max score
                            min_scores[i] = min(min_scores[i], score)  # Update min score

                # Print average first
                print('Average', end=' ')
                average_row = ['Average']
                for total in averages:
                    average_score = total / num_models
                    print('&', end=' ')
                    print(f'{average_score:.1f}', end=' ')
                    average_row.append(round(average_score, 1))
                print('\\\\')
                csv_data.append(average_row)

                # Print each model's scores
                for model in self._model_order:
                    if model in self._all_scores:
                        row_data = [model]  # Start with the model name
                        print(f"\\texttt{{{MODEL_NAME[model]}}}", end=' ')
                        for i, category in enumerate(categories):
                            print('&', end=' ')
                            score = self._all_scores[model][level][category]['g_pass']
                            # Highlight max and min scores
                            if score == max_scores[i]:
                                print(f'{{\\color{{green}}{score:.1f}}}', end=' ')
                            elif score == min_scores[i]:
                                print(f'{{\\color{{red}}{score:.1f}}}', end=' ')
                            else:
                                print(f'{score:.1f}', end=' ')  # Format to 1 decimal place
                            row_data.append(score)  # Add score to row data
                        print('\\\\')
                        csv_data.append(row_data)  # Add the row to CSV data

                if save_to_csv:
                    csv_file = os.path.join(self._save_dir, f'scores_of_level_{level}_shot{self._shot_num}.csv')
                    # Create DataFrame from collected CSV data
                    df = pd.DataFrame(csv_data, columns=['Model'] + categories)
                    df.to_csv(csv_file, index=False)
                                        
    @ensure_dir_exists
    @validate_level  
    def save_leaderboard_file(self, level: str) -> None:
        
        if not self._if_evaluated:
            self._evaluate()  

            
        # Create output directory if it doesn't exist
        output_dir = os.path.join(self._save_dir,f'leaderboard_{level}')
        os.makedirs(output_dir, exist_ok=True)
        
        # Get categories
        categories = list(self._all_scores[next(iter(self._all_scores))][level].keys())
        
        # Save each model's scores to a separate file
        for model in self._model_order:
            if model in self._all_scores:
                model_results = {
                    "config": {
                        "model_name": MODEL_ID[model],
                        # "model_type": "instruction-tuned",  # You may need to get this from your data
                        # "architecture": "None",  # You may need to get this from your data
                        # "weight_type": "Original",
                        "model_dtype": "bfloat16",
                        "model_sha": "main"
                        # "license": "Apache 2.0",
                        # "params": 0,  # You may need to get this from your data
                        # "likes": 0,  # You may need to get this from your data
                        # "still_on_hub": True,
                        # "revision": "None"  # You may need to get this from your data
                    },
                    "results": {}
                }
                
                # Store scores for each domain
                categories = list(self._all_scores[model][level].keys())
                for category in categories:
                    score = self._all_scores[model][level][category]['score']
                    model_results["results"][category] = {
                        "acc": round(score, 1)
                    }
                # Save to JSON file
                filename = os.path.join(output_dir, f'{model}.json')
                write_json(filename, model_results)
            
            
               
    @ensure_dir_exists
    @validate_level  
    def print_leaderboard_table(self, level: str) -> None:
        """
        Example:
            >>> evaluator.print_leaderboard_table('t1')
            | Model Name | Average Score |
            |------------|---------------|
            | GPT-4o    | 77.6          |
            | Claude-3.5-Sonnet | 76.2   |
            | Qwen2.5-3B-Instruct | 66.3 |
        """
        if not self._if_evaluated:
            self._evaluate()  

        data = self._all_scores
        models = list(data.keys())
        categories = list(data[models[0]][level].keys())[:-1]
        
        # Create a list of tuples (model_name, avg_score)
        scores = []
        for model in self._model_order:
            
            # model_name = MODEL_NAME[model]
            model_name = f"[{model}]({MODEL_INFO[model]['url']})"
            
            is_open = MODEL_INFO[model]['open']
            avg_score = data[model][level]['Average']['score']
            
            row = [model_name, is_open]
            
            for category in categories:
                score = data[model][level][category]['score']
                row.append(f"{score:.1f}")
            
            row.append(f"{avg_score:.1f}")
            
            scores.append(row)

        # Sort the scores by avg_score in descending order
        scores.sort(key=lambda x: x[-1], reverse=True)

        # Create the table header
        # table = "| Model Name | Open? |\n"
        header_list = ['Model', 'Open?']
        

        categories = list(self._all_scores[next(iter(self._all_scores))][level].keys())
        for category in categories:
            header_list.append(category)
        header_list.append('LexSafeBench')

        table = '|' + '|'.join(header_list) + '|\n'
        
        table +=  '|' + '------------|' * len(header_list) + '\n'
            
        # Populate the table with sorted scores
        for row in scores:
            table += "|" + "|".join(row) + "|\n"

        print(table)
    
    
    
    @ensure_dir_exists
    def print_elo_table(self) -> None:
        
        if not self._if_evaluated:
            self._evaluate()  
    
        # Print each model's scores
        for model in self._model_order:
            if model in self._all_scores:
                print(f"\\texttt{{{MODEL_NAME[model]}}}", end=' ')
                
                score = self._all_scores[model]['t1']['Average']['score']
                
                print(f"& {score:.1f} & {int(self._elo_rate[model])}\\\\")
                
    
    def statistics_categories(self, level: str, if_print: bool=True) -> None:
        
        if level not in self._all_stat: 
            evaluator = LexSafeModelEvaluator(self._result_path_list[0], self._map_dir_list, self._architecture_path, answer_type=self._answer_type, consider_refusal=self._consider_refusal)
            
            self._all_stat = {
                    't1': evaluator.first_level_count,
                    't2': evaluator.second_level_count,
                    't3': evaluator.third_level_count
                }
        
        if if_print:
            print(f"----------Task number for {level} level topic-----------")
            for k in self._all_stat[level]:
                print(f"{k}\t{self._all_stat[level][k]}")
                # print(f"{self._all_stat[level][k]}")

       
        csv_file = os.path.join(self._save_dir, f'stat_level_{level}.csv')
        df = pd.DataFrame(list(self._all_stat[level].items()), columns=['Category', 'Count'])
        df.to_csv(csv_file, index=False)

        return self._all_stat[level]

    def print_latex_statistics_categories(self, level: str) -> None:
        if level not in self._all_stat: 
            self.statistics_categories(level=level, if_print=False)

        categories = list(self._all_stat[level].keys())
        values = list(self._all_stat[level].values())
        
        header = "Category " + " & num \\\\"
        print(header)
        for i in range(len(categories)):
            print(f'{categories[i]} & {values[i]} \\\\')

        
    def plot_statistics_categories(self, level: str) -> None:
        if level not in self._all_stat: 
            self.statistics_categories(level=level, if_print=False)

        categories = list(self._all_stat[level].keys())
        values = list(self._all_stat[level].values())

        colors = []
        for i in range(len(categories)):
            if i < 3:
                colors.append('#E23D28') 
            elif i == 3 or i == 4:
                colors.append('#FF7D07')  
            elif i == 5 or i== 6 or i ==7:
                colors.append('#00688B')  
            else:
                colors.append('#008080') 

        plt.figure(figsize=(10, 6)) 
        plt.barh(categories, values, color=colors, edgecolor='none') 

        plt.title(f'Statistics for {level} Level Topic', fontsize=16)
        plt.ylabel('Categories', fontsize=14)
        plt.xlabel('Counts', fontsize=14)

        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)


        plt.box(False)
        plt.gca().invert_yaxis()

        plt.tight_layout()  

        plt.savefig(os.path.join(self._save_dir, f'num_stat_{level}.png'), dpi=300)  # 可以设置dpi
        plt.close()  
        
    @property
    def scores(self):
        if not self._if_evaluated:
            self._evaluate()  
        return self._all_scores


