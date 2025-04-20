import os
import hashlib
from tqdm import tqdm
from utils import write_json, load_json
from typing import List, Dict
from call_llm import get_response


from prompt import CONTENT_LABEL_SYSTEM_PROMPT, CONTENT_LABEL_USER_PROMPT

class SafeLabeler:
    def __init__(self, 
                 filepath: str, 
                 save_dir: str, 
                 architecture_path: str, 
                 classification_keys: List[str] = None, 
                 generate_id: bool = False, 
                 label_model: str = 'gpt-4o'):

        self._filepath = filepath
        self._dataname = os.path.splitext(os.path.basename(filepath))[0]
        self._data = load_json(self._filepath)

        if generate_id:
            self._generate_unique_id()

        self._save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self._label_path = os.path.join(save_dir, f'{self._dataname}_label_meta.json')
        print(self._label_path)
        self._map_path = os.path.join(save_dir, f'{self._dataname}_label_map.json')
        self._label_model = label_model
        self._classification_keys = classification_keys

        self._archi_path = architecture_path
        self._architecture = self._format_architecture()
        self._identifier = 'content'

        self._labeled_data = []
        self._label_map = {}

    def _generate_unique_id(self) -> None:
        m = hashlib.md5()
        for d in self._data:
            m.update(d['question'].encode('utf-8'))
            uid = m.hexdigest()
            d['id'] = uid

    def _format_architecture(self) -> str:
        archi_data = load_json(self._archi_path)
        structured_string = ""
        for category_number, (category, subcategories) in enumerate(archi_data.items(), start=1):
            structured_string += f"{category_number}. {category}\n"
            for subcategory_number, (subcategory, items) in enumerate(subcategories.items(), start=1):
                structured_string += f"    {category_number}.{subcategory_number} {subcategory}\n"
                for item_number, item in enumerate(items, start=1):
                    structured_string += f"        {category_number}.{subcategory_number}.{item_number} {item}\n"

        # print(structured_string)
        return structured_string

    def _parse_output(self, output_text: str) -> Dict[str, str]:
        try:
            result = {
                'first_level_topic': None,
                'second_level_topic': None,
                'third_level_topic': None
            }

            lines = [line.strip() for line in output_text.split('\n') if line.strip()]
            for line in lines:
                if '[[FIRST-LEVEL-TOPIC]]' in line:
                    result['first_level_topic'] = line.split(']]', 1)[1].strip()
                elif '[[SECOND-LEVEL-TOPIC]]' in line:
                    result['second_level_topic'] = line.split(']]', 1)[1].strip()
                elif '[[THIRD-LEVEL-TOPIC]]' in line:
                    result['third_level_topic'] = line.split(']]', 1)[1].strip()

            return result

        except Exception as e:
            raise ValueError(f"Failed to parse classification output: {str(e)}")

    def _label_to_map(self) -> None:
        for l in self._labeled_data:
            key = l[self._identifier]
            if key not in self._label_map:
                self._label_map[key] = {
                    'first_level_topic': l['first_level_topic'],
                    'second_level_topic': l['second_level_topic'],
                    'third_level_topic': l['third_level_topic']
                }

        write_json(self._map_path, self._label_map)

    def _label_by_content(self) -> None:

        if os.path.exists(self._label_path):
            
            self._labeled_data = load_json(self._label_path)
            existing_content = {item['content'] for item in self._labeled_data if item["first_level_topic"]}
            print(len(self._labeled_data))
            
        else:
            
            existing_content = set()

        
        print(len(self._data))
        # Get unique content
        content_list = []
        for d in self._data:
            
            c = ', '.join(d.get(k) for k in self._classification_keys if d.get(k)).strip()
            
            content_list.append(c)

        content_list = list(set(content_list) - existing_content)

        # Label new content
        for content in tqdm(content_list, desc="Labeling content"):
            output = get_response(CONTENT_LABEL_SYSTEM_PROMPT, CONTENT_LABEL_USER_PROMPT.format(content=content, architecture=self._architecture))
            label = self._parse_output(output)
            label['content'] = content
            self._labeled_data.append(label)
            # print('done')
            # print(len(self._labeled_data))
            write_json(self._label_path, self._labeled_data)

    def label(self) -> None:
        self._label_by_content()
        self._label_to_map()




l = SafeLabeler(
    filepath='',
    save_dir='',
    architecture_path='',
    # classification_keys=['l1', 'l2', 'question'], # for mc
    classification_keys=['l2', 'question'], # for open qa
    label_model='gpt-4o'
    # label_model='claude-3-5-sonnet-20241022'
    # label_model='gemini-1.5-pro'
    )    

l.label()