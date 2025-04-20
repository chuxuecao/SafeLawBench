import json

def load_json(file):
    with open(file,'r', encoding="utf8") as load_f:
        data = json.load(load_f)
        return data
    
def write_json(file, dict):
    with open(file, "w", encoding="utf8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)



model_order = [
 'gpt-4o',
 'claude-3-5-sonnet-20241022',
 'DeepSeek-V3',
 
 'Qwen2.5-3B-Instruct',
 'Qwen2.5-7B-Instruct',
 'Qwen2.5-14B-Instruct',
 'Qwen2.5-72B-Instruct',
 
 'glm-4-9b-chat',
 
 'gemma-2-2b-it',
 'gemma-2-27b-it',

 'vicuna-7b-v1.5',
 'vicuna-13b-v1.5',
 
 'Mistral-Small-Instruct-2409',
 'Mistral-Large-Instruct-2411',
 
 'Meta-Llama-3-8B-Instruct',
 'Meta-Llama-3-70B-Instruct',
 
 'Llama-3.1-8B-Instruct',
 'Llama-3.1-70B-Instruct',
#  'Multi-model-voting'
 ]


# value 对应图表中使用的模型名称
MODEL_NAME = {
    'gpt-4o': 'GPT-4o',
    'claude-3-5-sonnet-20241022': 'Claude-3.5-Sonnet',

    'Qwen2.5-3B-Instruct': 'Qwen2.5-3B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen2.5-7B-Instruct',
    'Qwen2.5-14B-Instruct': 'Qwen2.5-14B-Instruct',
    'Qwen2.5-72B-Instruct': 'Qwen2.5-72B-Instruct',
    
    'glm-4-9b-chat': 'GLM-4-9B-Chat',
    
    'gemma-2-2b-it': 'Gemma 2 IT 2B',
    'gemma-2-27b-it': 'Gemma 2 IT 27B',

    'vicuna-7b-v1.5': 'Vicuna-7B-V1.5',
    'vicuna-13b-v1.5': 'Vicuna-13B-V1.5',
    
    'Mistral-Small-Instruct-2409': 'Mistral-Small-Instruct',
    'Mistral-Large-Instruct-2411': 'Mistral-Large-Instruct',
    
    'Meta-Llama-3-8B-Instruct': 'Llama-3-8B-Instruct',
    'Meta-Llama-3-70B-Instruct': 'Llama-3-70B-Instruct',
    
    'Llama-3.1-8B-Instruct': 'Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'Llama-3.1-70B-Instruct',
}

# hugging face中的model id
MODEL_ID = {
    'gpt-4o': 'openai/gpt-4o',
    'claude-3-5-sonnet-20241022': 'anthropic/claude-3-5-sonnet-20241022',
    'DeepSeek-V3': 'deepseek-ai/DeepSeek-V3',

    'Qwen2.5-3B-Instruct': 'Qwen/Qwen2.5-3B-Instruct',
    'Qwen2.5-7B-Instruct': 'Qwen/Qwen2.5-7B-Instruct',
    'Qwen2.5-14B-Instruct': 'Qwen/Qwen2.5-14B-Instruct',
    'Qwen2.5-72B-Instruct': 'Qwen/Qwen2.5-72B-Instruct',
    
    'glm-4-9b-chat': 'THUDM/glm-4-9b-chat',
    
    'gemma-2-2b-it': 'google/gemma-2-2b-it',
    'gemma-2-27b-it': 'google/gemma-2-27b-it',

    'vicuna-7b-v1.5': 'lmsys/vicuna-7b-v1.5',
    'vicuna-13b-v1.5': 'lmsys/vicuna-13b-v1.5',
    
    'Mistral-Small-Instruct-2409': 'mistralai/Mistral-Small-Instruct-2409',
    'Mistral-Large-Instruct-2411': 'mistralai/Mistral-Large-Instruct-2411',
    
    'Meta-Llama-3-8B-Instruct': 'meta-llama/Llama-3-8B-Instruct',
    'Meta-Llama-3-70B-Instruct': 'meta-llama/Llama-3-70B-Instruct',
    
    'Llama-3.1-8B-Instruct': 'meta-llama/Llama-3.1-8B-Instruct',
    'Llama-3.1-70B-Instruct': 'meta-llama/Llama-3.1-70B-Instruct',
}



# leaderboard展示的模型信息
MODEL_INFO = {
    'gpt-4o': {
        'name': 'GPT-4o',
        'open': 'No',
        'url': 'https://openai.com/index/hello-gpt-4o/'
    },
    'claude-3-5-sonnet-20241022': {
        'name': 'Claude-3.5-Sonnet',
        'open': 'No',
        'url': 'https://www.anthropic.com/news/claude-3-5-sonnet'
    },
    'DeepSeek-V3': {
        'name': 'DeepSeek-V3',
        'open': 'Yes',
        'url': 'https://huggingface.co/deepseek-ai/DeepSeek-V3'
    },
    
    'Qwen2.5-3B-Instruct': {
        'name': 'Qwen2.5-3B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct'
    },
    'Qwen2.5-7B-Instruct': {
        'name': 'Qwen2.5-7B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'
    },
    'Qwen2.5-14B-Instruct': {
        'name': 'Qwen2.5-14B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct'
    },
    'Qwen2.5-72B-Instruct': {
        'name': 'Qwen2.5-72B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/Qwen/Qwen2.5-72B-Instruct'
    },
    'glm-4-9b-chat': {
        'name': 'GLM-4-9B-Chat',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/GLM-4-9B-Chat'
    },
    'gemma-2-2b-it': {
        'name': 'Gemma 2 IT 2B',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Gemma-2-2B-IT'
    },
    'gemma-2-27b-it': {
        'name': 'Gemma 2 IT 27B',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Gemma-2-27B-IT'
    },
    'vicuna-7b-v1.5': {
        'name': 'Vicuna-7B-V1.5',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Vicuna-7B-V1.5'
    },
    'vicuna-13b-v1.5': {
        'name': 'Vicuna-13B-V1.5',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Vicuna-13B-V1.5'
    },
    'Mistral-Small-Instruct-2409': {
        'name': 'Mistral-Small-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Mistral-Small-Instruct-2409'
    },
    'Mistral-Large-Instruct-2411': {
        'name': 'Mistral-Large-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Mistral-Large-Instruct-2411'
    },
    'Meta-Llama-3-8B-Instruct': {
        'name': 'Llama-3-8B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Llama-3-8B-Instruct'
    },
    'Meta-Llama-3-70B-Instruct': {
        'name': 'Llama-3-70B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Llama-3-70B-Instruct'
    },
    'Llama-3.1-8B-Instruct': {
        'name': 'Llama-3.1-8B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Llama-3.1-8B-Instruct'
    },
    'Llama-3.1-70B-Instruct': {
        'name': 'Llama-3.1-70B-Instruct',
        'open': 'Yes',
        'url': 'https://huggingface.co/models/Llama-3.1-70B-Instruct'
    },
}


import math
def calculate_variance_std(labels):
    mean = sum(labels) / len(labels)
    variance = sum((x - mean) ** 2 for x in labels) / len(labels)
    std_deviation = math.sqrt(variance)
    return round(std_deviation,2)