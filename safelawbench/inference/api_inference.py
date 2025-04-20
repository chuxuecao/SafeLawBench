import json
import os
import httpx
import asyncio
from tqdm.asyncio import tqdm
from utils import build_messages, judge_answer
import yaml
import argparse
import logging

from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default=None)
args = parser.parse_args()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

with open(args.config_file, 'r') as file:
    config = yaml.safe_load(file)

input_file = config['input_file']
output_dir = os.path.join(
    config['result_root_dir'], 
    config['dataset_version'],
    config['model_id'].split('/')[-1])

output_file = os.path.join(output_dir, f"{config['shot_num']}{config.get('save_id', '')}.json")

log_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
price_log_path = os.path.join(output_dir, 'price', f'log_{log_timestamp}.txt')

os.makedirs(os.path.join(output_dir, 'price'), exist_ok=True)

price_file = config.get('price_file')
with open(price_file, 'r') as p:
    price_data = json.load(p)

log_path = os.path.join(output_dir, 'log.txt')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
model_name = config['model_id']

price = price_data[model_name]


print(model_name)
print(config.get('shot_num'))
print(config.get('save_id'))

headers = {
    'Authorization': config['api_key'],
    'Content-Type': 'application/json',
}

CONCURRENCY = config.get('concurrency')
semaphore = asyncio.Semaphore(CONCURRENCY)

token_stats = {
    'total_prompt_tokens': 0,
    'total_completion_tokens': 0,
    'total_tokens': 0,
    'prompt_price': 0,
    'completion_price': 0,
    'total_price': 0,
    'requests_count': 0
}

def log_token_stats():
    with open(price_log_path, 'a') as log_file:
        log_file.write("\n" + "=" * 50 + "\n")
        log_file.write("SUMMARY STATISTICS\n")
        log_file.write(f"Total requests: {token_stats['requests_count']}\n")
        log_file.write(f"Total prompt tokens: {token_stats['total_prompt_tokens']}\n")
        log_file.write(f"Total completion tokens: {token_stats['total_completion_tokens']}\n")
        log_file.write(f"Total tokens: {token_stats['total_tokens']}\n")
        log_file.write(f"Total prompt price: {token_stats['prompt_price']}\n")
        log_file.write(f"Total completion price: {token_stats['completion_price']}\n")
        log_file.write(f"Total price: {token_stats['total_price']}\n")
        
        if token_stats['requests_count'] > 0:
            avg_prompt = token_stats['total_prompt_tokens'] / token_stats['requests_count']
            avg_completion = token_stats['total_completion_tokens'] / token_stats['requests_count']
            avg_total = token_stats['total_tokens'] / token_stats['requests_count']
            log_file.write(f"Average prompt tokens per request: {avg_prompt}\n")
            log_file.write(f"Average completion tokens per request: {avg_completion}\n")
            log_file.write(f"Average total tokens per request: {avg_total}\n")
            log_file.write(f"Average prompt price per request: {token_stats['prompt_price'] / token_stats['requests_count']}\n")
            log_file.write(f"Average completion price per request: {token_stats['completion_price'] / token_stats['requests_count']}\n")
            log_file.write(f"Average total price per request: {token_stats['total_price'] / token_stats['requests_count']}\n")

async def get_batch_response_with_retry(messages, n=1, max_retries=3):
    last_exception = None
    for attempt in range(max_retries):
        try:
            return await get_batch_response(messages, n)
        except Exception as e:
            last_exception = e
            logging.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(1 * (attempt + 1))
            else:
                logging.error(f"Failed after {max_retries} attempts")
                raise last_exception
            
async def get_batch_response(messages, n=1):
    all_answers = []
    async with semaphore:
        limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0), limits=limits) as client:
            tasks = []
            for message in messages:
                json_data = {
                    'model': model_name,
                    'messages': message,
                    'temperature': config.get('temperature', 1.0) + config.get('add_temperature', 0.0),
                    'stream': False,
                }
                task = asyncio.create_task(send_request(client, json_data, n))
                tasks.append(task)
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logging.error(f"Error in batch: {str(result)}")
                    all_answers.append(str(result))
                else:
                    all_answers.append(result)
    return all_answers

async def send_request(client, json_data, n):
    prompt_answers = []
    for _ in range(n):

        response = await client.post(config['base_url'], headers=headers, json=json_data)

        result = response.json()

        ans = result['choices'][0]['message']['content']
        
        prompt_tokens = result['usage']['prompt_tokens']
        completion_tokens = result['usage']['completion_tokens']
        total_tokens = result['usage']['total_tokens']
        
        prompt_price = prompt_tokens * price['prompt']
        completion_price = completion_tokens * price['completion']
        all_price = prompt_price + completion_price
        
        token_stats['total_prompt_tokens'] += prompt_tokens
        token_stats['total_completion_tokens'] += completion_tokens
        token_stats['total_tokens'] += total_tokens
        token_stats['prompt_price'] += prompt_price
        token_stats['completion_price'] += completion_price
        token_stats['total_price'] += all_price
        token_stats['requests_count'] += 1
        
        with open(price_log_path, 'a') as log_file:
            log_file.write(f"Request ID: {token_stats['requests_count']}\n")
            log_file.write(f"  Prompt tokens: {prompt_tokens}\n")
            log_file.write(f"  Completion tokens: {completion_tokens}\n")
            log_file.write(f"  Total tokens: {total_tokens}\n")
            log_file.write(f"  Prompt price: {prompt_price}\n")
            log_file.write(f"  Completion price: {completion_price}\n")
            log_file.write(f"  Total price: {all_price}\n")
            log_file.write("-" * 40 + "\n")
        
        prompt_answers.append(ans)

    return prompt_answers


def save_json(data, file_name):
    json_data = json.dumps(data, indent=2, ensure_ascii=False)
    with open(file_name, 'w') as file:
        file.write(json_data)
    file.close()

def split_into_batches(input_list, batch_size):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i:i + batch_size]

async def process_tasks(input_path):
    try:
        with open(input_path, 'r') as f:
            dataset = json.load(f)[config['start']: config['end']]
    except:
        with open(input_path, 'r') as f:
            dataset = json.load(f)
    
    current_dataset_ids = [d['id'] for d in dataset]

    with open(config['dev_path'], 'r') as f2:
        dev_data = json.load(f2)
    print('dev_data:', len(dev_data))

    try:
        with open(output_file, 'r') as f3:
            results = json.load(f3)
            logging.info(f'Loaded results: {len(results)}')
            
            results  = [d for d in results 
                            if d['generated_answer'] != '' and 'Expecting value: line 1 column 1 (char 0)' not in d['generated_answer']] 
            
            if config['task_type'] == 'multi-choice' or config['task_type'] == 'cot' or config['task_type'] == 'r-cot':
                
                
                results  = [d for d in results 
                            if d['generated_answer'][0] != '' and 
                            "'choices'" not in d['generated_answer'] and 
                            'score' in d and 
                            'All connection attempts failed' not in d['generated_answer'][0] 
                            and 'aiserver.v1.ErrorDetails' 
                            not in d['generated_answer'][0]
                            and '请求错误' not in d['generated_answer'][0]] #
            
        
            results  = [d for d in results if d['generated_answer'][0] != '' and d['generated_answer'] != '' and 'All connection attempts failed' not in d['generated_answer'] and 'Invalid type for url' not in d['generated_answer']] 
            
            results = [d for d in results if d['id'] in current_dataset_ids] 
            
            logging.info(f'Filtered results (valid data): {len(results)}')
            
    except:
        results = []
        logging.info('No existing results found')
    
    processed_id = [d['id'] for d in results ]
    
    logging.info(f'Total dataset size: {len(dataset)}')
    logging.info(f'Already processed: {len(processed_id)}')
        
    print(len(dataset))
    print('processed: ' , len(processed_id))
    
    dataset = [d for d in dataset if d['id'] not in processed_id]
    
    print(f'to be processed: {len(dataset)}')
    
    logging.info(f'To be processed: {len(dataset)}')
    
    messages = build_messages(dataset,
                              task_type=config['task_type'], 
                              shot_num=config['shot_num'],
                              dev_data=dev_data, 
                              output_format=config.get('output_format', 'direct'),
                              use_system_prompt=True,
                              thinking_starter=config.get('thinking_starter', None)) 

    tasks = []
    total_batches = len(list(split_into_batches(messages, config['batch_size'])))
    pbar = tqdm(total=total_batches, desc="Processing batches")

    for batch in split_into_batches(messages, config['batch_size']):
        task = asyncio.create_task(get_batch_response_with_retry(batch, n=config['n']))
        tasks.append(task)
    
    response_message_batches = await asyncio.gather(*tasks)

    for i, response_message_batch in enumerate(response_message_batches):
        pbar.update(1)  # 更新进度条
        start_index = i * config['batch_size']
        for j, answer in enumerate(response_message_batch):
            dataset_index = start_index + j
            if dataset_index < len(dataset):
                # print(answer)
                dataset[dataset_index]['generated_answer'] = answer
                
                if config['task_type'] == 'multi-choice' or config['task_type'] == 'cot' or config['task_type'] == 'r-cot':
                    dataset[dataset_index]['score'] = judge_answer(
                        # task_type=config['task_type'], 
                        correct_answer=dataset[dataset_index]['answer'], 
                        response=answer 
                        )


        results.extend(dataset[start_index:start_index + len(response_message_batch)])

        save_json(results, output_file)
            
    pbar.close()  
    
    # 记录最终统计信息
    log_token_stats()

    if config['task_type'] == 'multi-choice' or config['task_type'] == 'cot' or config['task_type'] == 'r-cot':  
        
        correct_num = len([d for d in results if d['score']])
        all_num = len(results)
        accuracy = round(100 * correct_num / all_num, 2)
    
    
        with open(log_path, "a") as log_file:
            shot_num = config['shot_num']
            task_type = config['task_type']
            log_file.write(f'api: task type: {task_type}\t shot num: {shot_num}\t  correct num: {correct_num}\t data num: {all_num}\t accuracy: {accuracy}\n')
    
    else:
        with open(log_path, "a") as log_file:
            task_type = config['task_type']
            log_file.write(f'api: task type: {task_type}\n')
        

async def main():
    
    with open(price_log_path, 'w') as log_file:
        log_file.write(f"Token statistics for model: {model_name}\n")
        log_file.write(f"Date: {datetime.now()}\n")
        log_file.write("=" * 50 + "\n\n")
        
        
    await process_tasks(input_file)
    
    
    
if __name__ == "__main__":
    
    # asyncio.run(main())
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
        
    with open(price_log_path, 'a') as log_file:
        log_file.write(f"Date: {datetime.now()}\n")