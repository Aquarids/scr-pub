import re
import json

@staticmethod
def extract_json(response: str) -> dict:
    pattern = r"```json\s*([\s\S]*?)\s*```"
    matches = re.findall(pattern, response)
    
    if not matches:
        pattern = r"({[\s\S]*})"
        matches = re.findall(pattern, response)
    
    if not matches:
        return None
        
    json_str = matches[0]
    
    try:
        json_str = ''.join(c for c in json_str if c.isprintable())
        try:
            return json.loads(json_str)
        except:
            pass
        json_str = ''.join(c for c in json_str if c.isprintable())
        json_str = json_str.replace(r'\"', '"').replace(r"\'", "'")
        json_str = json_str.replace("'", '"')
        json_str = re.sub(r'([{,]\s*)([^"\s][^:]*?)(:)', r'\1"\2"\3', json_str)
        json_str = re.sub(r'"\s*}\s*{?\s*"', '"},{"', json_str)
        json_str = re.sub(r'"\s*}\s*"', '"},"', json_str)
        json_str = re.sub(r']\s*"', '],"', json_str)
        return json.loads(json_str)
            
    except Exception as e:
        raise e

@staticmethod
def to_json(data):
    import json
    
    def json_serializable(obj):
        if isinstance(obj, set):
            return list(obj)
        else:
            try:
                return json.loads(json.dumps(obj))
            except:
                return str(obj)
    
    try:
        json_str = json.dumps(
            data,
            ensure_ascii=False,
            indent=4,
            default=json_serializable
        )
        
        formatted_str = (
            json_str
            .replace('    ', '  ')
            .replace('\\n', '\n')
            .replace('\\"', '"')
        )
        
        return formatted_str
        
    except Exception as e:
        raise e

@staticmethod
def compress_json(original_prompt):
    try:
        compressed = re.sub(
            r"```json\n([\s\S]*?)\n```",
            lambda m: "```json\n"
            + json.dumps(json.loads(m.group(1)), separators=(",", ":"))
            + "\n```",
            original_prompt,
        )

        compressed = re.sub(r"\n{3,}", "\n\n", compressed)
        compressed = re.sub(r"[ \t]{2,}", " ", compressed)
        return compressed.replace("\n\n", "\n")
    except Exception:
        return original_prompt
