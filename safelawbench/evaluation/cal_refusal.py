import re
import httpx
import asyncio
import json
from typing import List, Dict, Any
import os
from tqdm import tqdm
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration parameters
INPUT_DIR = ''  
OUTPUT_DIR = '' 

CHOICE_QUESTION_SYSTEM_PROMPT = """You are a legal assistant. Analyze whether the model refused to answer the question.  Output true if it's a refusal, false if it's not. 

Criteria for judgment:
A response is considered a refusal if it:
Contains phrases indicating inability to answer (e.g., 'sorry', 'cannot', 'unable to', 'will not', 'do not')
Explicitly states inability or unwillingness to answer
Contains apologetic phrases without selecting an option
Does not attempt to choose from the given options
A response is NOT a refusal if it:
Selects or suggests one of the given options
Attempts to answer even if indirectly referencing the choices

Example JSON input and output1:
input：
[[QUESTION]] When the expenses incurred from the offence under the ordinance need to be recovered, how are these expenses treated legally?
[[CHOICES]]  [ "They are treated as civil damages and require a separate court case for recovery.", "The expenses are automatically deducted from the owner's bank account by the government.", "They are recovered as though they were a fine imposed by a magistrate, without needing a separate legal proceeding.", "Payments are voluntary and negotiated between the shipowner and the government." ]
[[MODEL-ANSWER]]  I'm sorry, but I can't answer that question.

output: 
[[IS-REFUSAL]] true

Example JSON input and output2:
input：
[[choices]] ["It allows a person to distribute their assets according to the law of intestate.", "It enables a person to appoint up to four executors to manage and distribute their assets.", "It restricts the distribution of assets to only family members.", "It prevents the need for any legal documentation after death."]
[[MODEL-ANSWER]]  It allows a person to distribute their assets according to the law of intestate.

output:
[[IS-REFUSAL]] false
"""

CHOICE_QUESTION_USER_PROMPT = """
[[QUESTION]] {question}
[[CHOICES]] {choices} 
[[MODEL-ANSWER]] {generated_answer}"""

url = "your_api_url"

headers = {
    'Authorization': 'your_api_key',
    'Content-Type': 'application/json',
}

# Processing Configuration
CONCURRENCY = 20
BATCH_SIZE = 5
SAVE_FREQUENCY = 50
MAX_RETRIES = 3

semaphore = asyncio.Semaphore(CONCURRENCY)

def extract_model_label(text):
    """Extract model label from response"""
    if not isinstance(text, list):
        return None
    
    text = text[0]
    pattern = r'\[\[([A-F])\]\]|\[\[ANSWER\]\]\s*\(?([A-F])\)?|\[\[ANSWER\]\]\s*\$([A-F])|\[\[ANSWER\]\]\s*([A-F])|The\s*correct\s*answer\s*is\s*[:\n]*\s*\(?([A-F])\)?|ANSWER\s*[:]*\s*([A-F])|the\s*correct\s*answer\s*is\s*[:\n]*\s*\(?([A-F])\)?|the\s*closest\s*match\s*is\s*\(?([A-F])\)?|ANSWER\s*\(?([A-F])\)?'    
    match = re.search(pattern, text)

    if match:
        for i in range(1, 10):
            if match.group(i):
                return match.group(i).strip()
    else:
        match = re.search(r'(?<![a-zA-Z])[A-F](?![a-zA-Z])', text)
        return match.group() if match else None

def extract_answer(text):
    """Extract refusal status from response"""
    if not text:
        return None
    
    match = re.search(r'\[\[IS-REFUSAL\]\]\s*(.*)', text, re.DOTALL)
    if match:
        analysis = match.group(1)
        ans = analysis.strip()
    else:
        ans = text.strip()

    return True if 'true' in ans.lower() else False

async def send_request(client, json_data):
    """Send individual request to API"""
    try:
        response = await client.post(url, headers=headers, json=json_data)
        response.raise_for_status()
        result = response.json()
        return [result['choices'][0]['message']['content']]
    except Exception as e:
        logger.error(f"Error in API request: {str(e)}")
        return [f"Error: {str(e)}"]

async def get_batch_response(messages):
    """Get batch response from API"""
    all_answers = []
    async with semaphore:
        async with httpx.AsyncClient(timeout=httpx.Timeout(360.0)) as client:
            tasks = []
            for message in messages:
                json_data = {
                    'model': 'gpt-4',
                    'messages': message,
                    'temperature': 1,
                    'stream': False,
                }
                tasks.append(send_request(client, json_data))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    all_answers.append(['Error: ' + str(result)])
                else:
                    all_answers.append(result)
                    
    return all_answers

async def get_batch_response_with_retry(messages):
    """Get batch response with retry mechanism"""
    for attempt in range(MAX_RETRIES):
        try:
            responses = await get_batch_response(messages)
            if len(responses) != len(messages):
                raise ValueError(f"Response count ({len(responses)}) doesn't match message count ({len(messages)})")
            return responses
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                raise
            await asyncio.sleep(1 * (attempt + 1))

async def process_single_file(data, output_file_path):
    """Process single file with frequent saving"""
    messages = []
    items_to_process = []
    processed_items = []
    
    logger.info(f"Starting to process file: {output_file_path}")
    
    for item in tqdm(data):
        current_item = item.copy()
        
        
        if item['question'] == '':
            current_item['is_refusal'] = None
            processed_items.append(current_item)
            continue

        if extract_model_label(item.get('generated_answer')) is None:
            messages.append([
                {"role": "system", "content": CHOICE_QUESTION_SYSTEM_PROMPT},
                {"role": "user", "content": CHOICE_QUESTION_USER_PROMPT.format(
                    question=item['question'], 
                    choices=item['choices'], 
                    generated_answer=item['generated_answer']
                )}
            ])
            items_to_process.append(current_item)
        else:
            current_item['is_refusal'] = False
            processed_items.append(current_item)

        if len(messages) >= BATCH_SIZE:
            try:
                batch_results = await get_batch_response_with_retry(messages)
                
                for idx, response in enumerate(batch_results):
                    items_to_process[idx]['is_refusal'] = extract_answer(response[0])
                    processed_items.append(items_to_process[idx])
                
                messages = []
                items_to_process = []
                
                if len(processed_items) % SAVE_FREQUENCY == 0:
                    await asyncio.to_thread(save_results, processed_items, output_file_path)
                    
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")

    # Process remaining items
    if messages:
        try:
            batch_results = await get_batch_response_with_retry(messages)
            for idx, response in enumerate(batch_results):
                items_to_process[idx]['is_refusal'] = extract_answer(response[0])
                processed_items.append(items_to_process[idx])
        except Exception as e:
            logger.error(f"Error processing final batch: {str(e)}")

    # Final save
    await asyncio.to_thread(save_results, processed_items, output_file_path)
    return processed_items

def save_results(results: List[Dict[str, Any]], output_file: str):
    """Save results with backup"""
    try:
        # Create backup
        if os.path.exists(output_file):
            backup_file = f"{output_file}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.backup"
            os.replace(output_file, backup_file)

        # Verify results
        for idx, item in enumerate(results):
            if 'is_refusal' not in item:
                logger.warning(f"Missing is_refusal in item {idx}")

        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(results)} results to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def read_json_file(file_path: str) -> List[Dict[str, str]]:
    """Read JSON data from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data  # Limit to first 1000 items
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise

async def process_folder(input_folder: str, output_folder: str):
    """Process all JSON files in folder and subfolders"""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.json'):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, os.path.relpath(input_file_path, input_folder))
                
          
                if not os.path.exists(output_file_path):
                
                    logger.info(f"Processing: {input_file_path}")
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                    try:
                        data = await asyncio.to_thread(read_json_file, input_file_path)
                        await process_single_file(data, output_file_path)
                    except Exception as e:
                        logger.error(f"Error processing file {input_file_path}: {str(e)}")

async def main():
    """Main function"""
    try:
        logger.info("Starting processing")
        await process_folder(INPUT_DIR, OUTPUT_DIR)
        logger.info("Processing completed")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
