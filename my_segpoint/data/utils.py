import os, sys
import msgspec

def load_json(json_path: str):
    with open(json_path, 'r') as j:
        json_data = msgspec.json.decode(j.read())

    return json_data