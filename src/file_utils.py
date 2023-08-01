import codecs
import os
from pathlib import Path
import shutil
import json

def read_file(file_path):
    with open(file_path, encoding='utf-8', errors='ignore') as f:
        return f.readlines()

def read_json_file(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data

def write_list_to_file(file_path, content):
    with open(file_path, 'w') as f:
        for p in content:
            f.write(p)
            f.write('\n')

def write_string_to_file(file_path, content):
    with open(file_path, 'w') as f:
        f.write(content)

def delete_dirs(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def create_dirs(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def file_exists(file_path):
    return os.path.exists(file_path)

def dir_exists(file_path):
    return os.path.exists(file_path)
def get_files_in_directory(dir_path, extensions):
    files = os.listdir(dir_path)

    valid_files = []
    for f in files:
        if f =='.DS_Store':
            continue
        for extension in extensions:
            if f.endswith(extension):
                valid_files.append(f)
                break
        else:
            valid_files.append(f)

    return valid_files