
import re
import argparse
import yaml
import os

from typing import Any, Dict, List
from transformers import AutoTokenizer
import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

from utils import build_messages, load_json, write_json, judge_answer, extract_answer, evaluate, calculate_variance_std, calculate_difference, calculate_mean_deviation

assert Version(ray.__version__) >= Version(
"2.22.0"), "Ray version must be at least 2.22.0"

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None)
args = parser.parse_args()

with open(os.path.join(args.config_file), 'r') as file:
    config = yaml.safe_load(file)
    
sampling_params = SamplingParams(
    # use_beam_search = False,
    # # do_sample=True,
    n = config.get('n', 1), # for multi answer
    # best_of = config.get('best_of', 1), # for multi answer
    temperature=config.get('temperature', 1.0) + config.get('add_temperature', 0.0), 
    top_p=config.get('top_p',1.0),
    min_p=config.get('min_p', 0.0),
    top_k=config.get('top_k', -1),
    presence_penalty=config.get('presence_penalty', 0.0),
    repetition_penalty = config.get('repetition_penalty', 1.0),
    
    max_tokens=config['max_token'])

# Set tensor parallelism per instance.
tensor_parallel_size = config['tensor_parallel_size']

# Set number of instances. Each instance will use tensor_parallel_size GPUs.
num_instances = config['num_instances']

def scheduling_strategy_fn():
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 20
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))

    
class LLMPredictor:
    def __init__(self):
        model_id = config['model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir = config['cache_dir'],
            trust_remote_code=True
            )
        
        self.llm = LLM(model=model_id,
                       tensor_parallel_size=tensor_parallel_size,
                       gpu_memory_utilization=0.9,
                       max_model_len=config['max_model_len'],
                       trust_remote_code=True
                       )
        self._dev_data = load_json(config['dev_data_dir'])
        
    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        
        messages = build_messages(batch['data'], 
                                  task_type = config['task_type'], 
                                  shot_num = config["shot_num"], 
                                  dev_data = self._dev_data, 
                                  output_format=config.get('output_format', 'direct'),
                                  use_system_prompt = config['use_system_prompt'], 
                                  thinking_starter=config.get('thinking_starter', None))
        
        prompts = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        
        outputs = self.llm.generate(prompts, sampling_params)
        generated_text: List[str] = []
        # generated_text = [output.outputs[0].text for output in outputs] 
        generated_text = [[output.outputs[i].text for i in range(config.get('n', 1))] for output in outputs]
        
        if config['clean_output']:
            generated_text = [extract_answer(a) for a in generated_text]
        
        return {"generated_text": generated_text}

class VLLMInference:
    def __init__(self):
        input_file = config['input_file']
        output_dir = os.path.join(
            config['result_root_dir'], 
            config['dataset_version'],
            config['model_id'].split('/')[-1])

        self.output_file = os.path.join(output_dir, f"{config.get('shot_num')}{config.get('save_id', '')}.json")
        
        os.makedirs(output_dir, exist_ok=True)
            
        self.log_path = os.path.join(output_dir, 'log.txt')
        try:
            self.json_ds = load_json(input_file)[config['start']:config['end']]
        except:
            self.json_ds = load_json(input_file)
        self.ds = ray.data.from_numpy(np.array(self.json_ds))
        self.output = None
        self.processed_output = None

        self.resources_kwarg: Dict[str, Any] = {}
        if tensor_parallel_size == 1:
            self.resources_kwarg["num_gpus"] = 1
        else:
            self.resources_kwarg["num_gpus"] = 0
            self.resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

    def _inference(self):
        self.ds = self.ds.map_batches(
            LLMPredictor,
            concurrency=num_instances,
            batch_size=config['batch_size'],
            **self.resources_kwarg,
        )
        self.outputs = self.ds.take_all()

    def _process_results(self):
        self.processed_output = []
        for i, output in enumerate(self.outputs):
            one_result = self.json_ds[i]
            answer = output['generated_text']
            one_result['generated_answer'] = answer
            if config['task_type'] == 'multi-choice':
                # one_result['scores'] = [1 if judge_answer(one_result['answer'], i) else 0 for i in answer ]
                one_result['score'] = judge_answer(one_result['answer'], answer[0])
                
                # one_result['std_variance'] = calculate_variance_std(one_result['scores']) if len(answer)>1 else 0
                # one_result['difference'] = calculate_difference(one_result['scores']) if len(answer)>1 else 0
                # one_result['mean_difference'] = calculate_mean_deviation(one_result['scores']) if len(answer)>1 else 0
            
            self.processed_output.append(one_result)
        
        # if config['task_type'] == 'multi-choice':
        #     cnt_correct, cnt_all, accuracy = evaluate(self.processed_output)
        #     with open(self.log_path, "a") as log_file:
        #         log_file.write(f'\t correct num: {cnt_correct}\t data num: {cnt_all}\t accuracy: {accuracy}\n')
    
        write_json(self.output_file, self.processed_output)
    
    def run_inference(self):
        self._inference()
        self._process_results()



ifr = VLLMInference()
ifr.run_inference()