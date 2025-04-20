import json

def load_json(file):
    with open(file,'r', encoding="utf8") as load_f:
        data = json.load(load_f)
        return data
    
def write_json(file, dict):
    with open(file, "w", encoding="utf8") as f:
        json.dump(dict, f, indent=4, ensure_ascii=False)
