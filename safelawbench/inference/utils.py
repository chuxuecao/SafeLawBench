import json
import re
import os
from prompt import MC_HK_QUESTION_SYSTEM_PROMPT, MC_MAINLAND_CHINA_QUESTION_SYSTEM_PROMPT, MC_QUESTION_USER_PROMPT, OPEN_QUESTION_SYSTEM_PROMPT, OPEN_QUESTION_USER_PROMPT, COT_OUTPUT_FORMAT, DIRECT_OUTPUT_FORMAT

def load_json(file):
    with open(file,'r', encoding="utf8") as load_f:
        data = json.load(load_f)
        return data
    
def write_json(file, dict):
    with open(file, "w", encoding="utf8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)

def build_mc_example(d, id=None, with_answer = False):
    question = d['question']
    choices = d['choices']
    answer = d['answer']
    letters =['A', 'B', 'C', 'D', 'E', 'F']
    choices_str = ''
    
    for i in range(len(choices)):
        choices_str +=  f"\n({letters[i]}) {choices[i]}"
    
   
    enquiry = f'[[QUESTION]] {question}{choices_str}'
    
    if with_answer:
        example = f'Example {id}:\n{enquiry}\n[[ANSWER]] ${answer}'
    else:
        example = f'{enquiry}\n[[ANSWER]]'
        
    return example
    
def build_user_prompts(data, task_type, shot_num, dev_data):
    
    if task_type == 'multi-choice':
        if shot_num > 0:
            dev_data_example = dev_data[0:shot_num]
            # examples = [build_mc_example(d, with_answer=True) for d in dev_data_example]
            examples = []
            for i in range(len(dev_data_example)):
                examples.append(build_mc_example(dev_data_example[i], i+1, with_answer=True))
            
            example_prompt = '\n\n'.join(examples)
            few_shot_prompt = f'Here are some examples:\n{example_prompt}\n\nNow answer the following question:\n'
        else:
            few_shot_prompt = ''
        
        prompts = []
        for d in data:
            prompt = few_shot_prompt + MC_QUESTION_USER_PROMPT.format(question = build_mc_example(d))
            prompts.append(prompt)
            
    elif task_type == 'open-qa':
        prompts =[OPEN_QUESTION_USER_PROMPT.format(question = d['question']) for d in data]
      
    return prompts




def build_messages(data, task_type, shot_num, dev_data, output_format = 'direct', use_system_prompt = True, thinking_starter=None):
    
    # print(task_type)
    
    
    user_prompts = build_user_prompts(data, task_type, shot_num, dev_data)
    
    
    messages = []
    
    if task_type == 'multi-choice':
      
        if use_system_prompt:
            for i in range(len(data)):
                
                system_prompt = MC_HK_QUESTION_SYSTEM_PROMPT if data[i]['region'] == 'Hong Kong' else MC_MAINLAND_CHINA_QUESTION_SYSTEM_PROMPT
                
                if output_format == 'direct':
                    system_prompt = system_prompt.format(format_specification=DIRECT_OUTPUT_FORMAT) 
                elif output_format == 'cot':
                    system_prompt = system_prompt.format(format_specification = COT_OUTPUT_FORMAT)
                elif output_format == 'empty':
                    system_prompt = system_prompt.format(format_specification = '')
                else:
                    print("Output format not supported!")
                
                messages.append([
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f'{user_prompts[i]}{thinking_starter if thinking_starter else ''}'}
                        ]
                )
                
                
        else:
            for i in range(len(data)):
                system_prompt = MC_HK_QUESTION_SYSTEM_PROMPT if data[i]['region'] == 'Hong Kong' else MC_MAINLAND_CHINA_QUESTION_SYSTEM_PROMPT
                
                if output_format == 'direct':
                    system_prompt = system_prompt.format(format_specification=DIRECT_OUTPUT_FORMAT) 
                elif output_format == 'cot':
                    system_prompt = system_prompt.format(format_specification = COT_OUTPUT_FORMAT)
                elif output_format == 'empty':
                    system_prompt = system_prompt.format(format_specification = '')
                else:
                    print("Output format not supported!")
                
                messages.append([
                        {'role': 'user', 'content': f'{system_prompt}\n{user_prompts[i]}{thinking_starter if thinking_starter else ''}'}
                    ]
                )
    elif task_type == 'open-qa':
        
        system_prompt = OPEN_QUESTION_SYSTEM_PROMPT
        
        if use_system_prompt:
            for i in range(len(data)):
                
                messages.append([
                            {'role': 'system', 'content': system_prompt},
                            {'role': 'user', 'content': f'{user_prompts[i]}{thinking_starter if thinking_starter else ''}'}
                        ]
                )
        else:
            for i in range(len(data)):
                messages.append([
                        {'role': 'user', 'content': f'{system_prompt}\n{user_prompts[i]}{thinking_starter if thinking_starter else ''}'}
                    ]
                )
        print(messages[0])
    return messages



def extract_label(text, task_type):
    
    
    if task_type == 'multi-choice':
        pattern = r'\[\[([A-F])\]\]|\[\[ANSWER\]\]\s*\(?([A-F])\)?|\[\[ANSWER\]\]\s*\$([A-F])|\[\[ANSWER\]\]\s*([A-F])|The\s*correct\s*answer\s*is\s*[:\n]*\s*\(?([A-F])\)?|ANSWER\s*[:]*\s*([A-F])|the\s*correct\s*answer\s*is\s*[:\n]*\s*\(?([A-F])\)?|the\s*closest\s*match\s*is\s*\(?([A-F])\)?|ANSWER\s*\(?([A-F])\)?'    
        match = re.search(pattern, text)

        if match:
            for i in range(1, 10):
                if match.group(i):
                    return match.group(i).strip() 
        else:
            # 目前补充的实验是cot所以这里先把调转顺序写死
            text = text[::-1]
            
            match = re.search(r'(?<![a-zA-Z])[A-F](?![a-zA-Z])', text)
            if match:
                return match.group()
            
            else:
                return text.strip()
    
    else:
        return text.strip()

    
def judge_answer(correct_answer, response):

    try:
        response = response[0]
    except:
        pass
    
    if correct_answer.strip() == response.strip():
        return True
    
    else:
        return extract_label(response, task_type='multi-choice') == correct_answer.strip()

def extract_answer(text):
    """
    for gemma
    """
    try:
        try:
            pattern = r'<start_of_turn>model(.*?)<end_of_turn>'  
            matches = re.findall(pattern, text, re.DOTALL)  
            return matches[0]
        except:
            pattern = r'<end_of_turn>(.*)' 
            match = re.search(pattern, text, re.DOTALL) 

            if match:
                return match.group(1).strip()  
            else:
                return text
    except:
        return text

def evaluate(data):
    correct_data = [d for d in data if d['scores'][0]]
    
    return len(correct_data), len(data), round(100*len(correct_data)/len(data), 1)

import math
def calculate_variance_std(labels):
    mean = sum(labels) / len(labels)
    variance = sum((x - mean) ** 2 for x in labels) / len(labels)
    std_deviation = math.sqrt(variance)
    return round(std_deviation,2)

def calculate_difference(labels):
    differences = [abs(labels[i+1] - labels[i]) for i in range(len(labels)-1)]
    mean_difference = sum(differences) / len(differences)
    return round(mean_difference, 2)

def calculate_mean_deviation(labels):
    mean = sum(labels) / len(labels)
    mean_deviation = sum(abs(x - mean) for x in labels) / len(labels)
    return round(mean_deviation, 2)


def build_evaluation_messages(data, use_system_prompt=True):
    user_prompts = build_elo_prompts(data)
    messages = [] 
    if use_system_prompt:
        for i in range(len(data)):
            messages.append([
                        {'role': 'system', 'content':  OPEN_QA_EVALUATION_SYSTEM_PROMPT},
                        {'role': 'user', 'content': user_prompts[i]}
                    ]
            )
    else:
        for i in range(len(data)):
            messages.append([
                    {'role': 'user', 'content': f'{ OPEN_QA_EVALUATION_SYSTEM_PROMPT}\n{user_prompts[i]}'}
                ]
            )
    return messages


def build_elo_prompts(data):
    prompts =[OPEN_QA_EVALUATION_USER_PROMPT.format(question = d['question'],ground_truth_answer = d['answer'],model1_answer = d['model1_answer'], model2_answer = d['model2_answer']) for d in data]
    return prompts

def parse_evaluation_response(response_text):
    m1 = 0
    m2 = 0
    pattern = r'(?:\[\[BEST-MODEL\]\]\s*)?(model1|model2)'
    for i in response_text:
        match = re.search(pattern, i, re.IGNORECASE)
    if match:
        if match.group(1).lower() == 'model1':
            m1 += 1
        else:
            m2 += 1
    if m1 >= m2:
        return "model1"
    else:
        return "model2"

import json

def merge_data(data1, data2, output_file): 
    merged_data = []
    for i in range(len(data1)):
        combine = data1[i]
        combine['model1_answer'] = data1[i]['generated_answer'][0][11:]
        del combine['generated_answer']
        if data1[i]['id'] == data2[i]['id']:
            combine['model2_answer'] = data2[i]['generated_answer'][0][11:]
        else:
            for item in data2:
                if item['id'] == data1[i]['id']:
                    combine['model2_answer'] = data2[i]['generated_answer']
                else:
                    combine['model2_answer'] = ""
        merged_data.append(combine)
    return merged_data
