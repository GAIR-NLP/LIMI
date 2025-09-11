import json
import os
import random
from pathlib import Path

def process_json_files():

    source_dir = Path("pr_data/task_400")
    target_dir = Path("pr_data/task_final_400")

    target_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(source_dir.glob("*.json"))
    
    if not json_files:
        print(f"no json in {source_dir} ")
        return
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if "modified_files" not in data:
                print(f"jump {json_file.name}: no modified files")
                skipped_count += 1
                continue
            
            modified_files = data["modified_files"]
            
            has_md_file = any('.md' in file for file in modified_files)
            
            if has_md_file:
                print(f"jump{json_file.name}: modified_files including .md 文件")
                skipped_count += 1
                continue

            if "test_query" not in data:
                print(f"jump {json_file.name}: lose test_query")
                skipped_count += 1
                continue
            
            test_query = data["test_query"]

            if len(modified_files) == 1:
                files_text = f"the {modified_files[0]}"
            else:
                files_text = f"these files: {', '.join(modified_files)}"

            test_query += f" You only need to modify {files_text} to complete the task."

            if random.random() < 0.4:
                test_query += " You are allowed to install any necessary environments when completing the task."
            else: 
                test_query += " You are NOT allowed to install any necessary environments when completing the task"

            data["test_query"] = test_query

            target_file = target_dir / json_file.name
            with open(target_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            processed_count += 1
            print(f"process: {json_file.name}")
            
        except Exception as e:
            print(f"process {json_file.name} error: {str(e)}")
            skipped_count += 1
    
    print(f"\nfinish!")


if __name__ == "__main__":
    random.seed(42)
    
    process_json_files()