import json
import os
import sys
sys.path.append(os.getcwd())
import requests
import argparse

from api_call.api_call import create_chat_completion
from pr.util.final_repo_query import query_repo_stream
from pr.pr2task.task_meta import TASKS, GENERATION_TASKS
from tqdm import tqdm

# threhold = 400
threhold = 2000


def generate_task_description(repo_id, pr_number):
    repo_name = query_repo_stream("pr_data/star_5_final.json", repo_id)
    if not repo_name:
        repo_name = "unknow"
        return None

    repo_url = f"https://github.com/{repo_name}.git"
    repo_pr_file_path = f"pr_data/pr_data_{threhold}/enhanced_data_{repo_id}_{pr_number}.jsonl"


    with open(repo_pr_file_path, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
        if not first_line:
            return None
        try:
            pr_data = json.loads(first_line)
        except json.JSONDecodeError:
            return None

    # filter pr_data to only keep necessary fields
    # temporarily, we do not filter, because we want to give the model more context

    input = GENERATION_TASKS.format(
        PR_DATA=json.dumps(pr_data, ensure_ascii=False, indent=2)
    )

    user_message = {
        "role": "user",
        "content": input,
    }

    messages = [user_message]
    task_description = create_chat_completion(messages, model="gpt-5")

    if task_description is None:
        return None

    patch = []
    # get patch content
    for segment in pr_data.get("segments", []):
        if "files" not in segment:
            continue
        for file_patch in segment["files"]:
            patch.append(file_patch)

    if task_description is not None and "```json" in task_description:
        task_description = task_description.strip().replace("```json", "")
    if task_description is not None and "```" in task_description:
        task_description = task_description.replace("```", "")
    
    head_sha, base_sha = fetch_checksome(repo_name, pr_number)
    if head_sha is None or base_sha is None:
        print("cannot obtain PR.")
        return None

    # check if task_description is valid json
    try:
        task_description_data = json.loads(task_description)
        # output_path = f"./pr/data/test_processed/task_descriptions_{repo_id}_{pr_number}.json"
        output_path = f"pr/data/pr_data_task_{threhold}/task_descriptions_{repo_id}_{pr_number}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        task_description_data["repo_url"] = repo_url.removesuffix(".git")
        task_description_data["patch"] = patch
        task_description_data["head_sha"] = head_sha
        task_description_data["base_sha"] = base_sha
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(task_description_data, f, ensure_ascii=False, indent=2)
    except json.JSONDecodeError as e:
        print(f"Warning: Not valid sample {e}")
        output_path = f"./pr_data/tasks/task_{threhold}/task_descriptions_{repo_id}_{pr_number}.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(task_description)
        return None

    return task_description


def fetch_checksome(repo_name, pr_number):

    api_url=f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    response = requests.get(api_url)
    if response.status_code == 200:
        pr_data = response.json()
        return pr_data.get("head", {}).get("sha", "unknown"),pr_data.get("base", {}).get("sha", "unknown")
    return None, None


def args_parser():
    parser = argparse.ArgumentParser(description="generate tasks")
    parser.add_argument("--repo_id", type=str, required=True, help="repo ID")
    parser.add_argument("--pr_number", type=int, required=True, help="PR id")
    return parser.parse_args()


if __name__ == "__main__":
    folder=f"pr_data/pr_data_{threhold}"
    files=os.listdir(folder)
    n=6
    current=0
    for file in tqdm(files):
        if "enhanced_data" in file and current<=n:
            file_path = os.path.join(folder, file)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                target_string = '"relevance_type": "changed_file"'
                count = file_content.count(target_string)
                
                if count != 1:
                    print(f"jump {file}，'relevance_type': 'changed_file' occur {count} times")
                    continue
                    
            except Exception as e:
                print(f"Read {file} Error: {e}")
                continue
            
            parts=file.replace("enhanced_data_","").replace(".jsonl","").split("_")
            if len(parts)!=2:
                continue
            repo_id=parts[0]
            pr_number=parts[1]
            print(f"Process {repo_id} {pr_number}")
            description = generate_task_description(repo_id, pr_number)
            
            print("finished")
            print("=="*20)







# import json
# import os
# import sys
# sys.path.append(os.getcwd())
# import requests
# import argparse

# from api_call.api_call import create_chat_completion
# from pr.util.final_repo_query import query_repo_stream
# from pr.pr2task.task_meta import TASKS, GENERATION_TASKS



# def generate_task_description(repo_id, pr_number):
#     """
#     生成任务描述

#     Args:
#         repo_id (str): 仓库ID
#         pr_number (int): PR编号

#     Returns:
#         str: 任务描述
#     """
#     # 查询repo_name
#     repo_name = query_repo_stream("./pr/util/star_5_final.json", repo_id)
#     if not repo_name:
#         repo_name = "未知仓库"
#         return None

#     repo_url = f"https://github.com/{repo_name}.git"
#     repo_pr_file_path = f"/inspire/hdd/project/qproject-fundationmodel/public/mhjiang/sii-cli-model/pr/data/test/enhanced_data_{repo_id}_{pr_number}.jsonl"

#     # # 如果任务描述已存在，跳过生成
#     # output_path = f"./pr/pr_data_task/task_descriptions_{repo_id}_{pr_number}.json"
#     # if os.path.exists(output_path):
#     #     print(f"任务描述已存在，跳过生成: {output_path}")
#     #     return None

#     with open(repo_pr_file_path, "r", encoding="utf-8") as f:
#         first_line = f.readline().strip()
#         if not first_line:
#             return None
#         try:
#             pr_data = json.loads(first_line)
#         except json.JSONDecodeError:
#             return None

#     # filter pr_data to only keep necessary fields
#     # temporarily, we do not filter, because we want to give the model more context

#     input = GENERATION_TASKS.format(
#         PR_DATA=json.dumps(pr_data, ensure_ascii=False, indent=2)
#     )

#     user_message = {
#         "role": "user",
#         "content": input,
#     }

#     messages = [user_message]

#     # 调用API生成任务描述
#     task_description = create_chat_completion(messages, model="gpt-5")

#     if task_description is None:
#         return None

#     patch = []
#     # get patch content
#     for segment in pr_data.get("segments", []):
#         if "files" not in segment:
#             continue
#         for file_patch in segment["files"]:
#             patch.append(file_patch)

#     if task_description is not None and "```json" in task_description:
#         task_description = task_description.strip().replace("```json", "")
#     if task_description is not None and "```" in task_description:
#         task_description = task_description.replace("```", "")
    
#     head_sha, base_sha = fetch_checksome(repo_name, pr_number)
#     if head_sha is None or base_sha is None:
#         print("无法获取PR的checksum信息。")
#         return None

#     # check if task_description is valid json
#     try:
#         task_description_data = json.loads(task_description)
#         # output_path = f"./pr/data/test_processed/task_descriptions_{repo_id}_{pr_number}.json"
#         output_path = f"./pr/data/pr_data_task_1800/task_descriptions_{repo_id}_{pr_number}.json"
#         task_description_data["repo_url"] = repo_url
#         task_description_data["patch"] = patch
#         task_description_data["head_sha"] = head_sha
#         task_description_data["base_sha"] = base_sha
#         with open(output_path, "w", encoding="utf-8") as f:
#             json.dump(task_description_data, f, ensure_ascii=False, indent=2)
#     except json.JSONDecodeError as e:
#         print(f"警告: 生成的任务描述不是有效的JSON格式: {e}")
#         output_path = f"./pr_data_task_1800/task_descriptions_{repo_id}_{pr_number}.txt"
#         with open(output_path, "w", encoding="utf-8") as f:
#             f.write(task_description)
#         return None

#     return task_description


# def fetch_checksome(repo_name, pr_number):
#     """
#     获取PR的checksum

#     Args:
#         repo_name (str): 仓库名称
#         pr_number (int): PR编号

#     Returns:
#         str: PR的checksum
#     """
#     # 这里是获取checksum的逻辑
#     # 例如，可以通过API调用获取PR的详细信息，然后计算checksum
#     api_url=f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
#     response = requests.get(api_url)
#     if response.status_code == 200:
#         pr_data = response.json()
#         # 计算checksum
#         return pr_data.get("head", {}).get("sha", "unknown"),pr_data.get("base", {}).get("sha", "unknown")
#     return None, None


# def args_parser():
#     parser = argparse.ArgumentParser(description="生成PR任务描述")
#     parser.add_argument("--repo_id", type=str, required=True, help="仓库ID")
#     parser.add_argument("--pr_number", type=int, required=True, help="PR编号")
#     return parser.parse_args()


# if __name__ == "__main__":
#     # args = args_parser()
#     # repo_id = args.repo_id
#     # pr_number = args.pr_number
#     # description = generate_task_description(repo_id, pr_number)
#     # print("生成的任务描述:")
#     # folder="/inspire/hdd/project/qproject-fundationmodel/public/mhjiang/sii-cli-model/pr/data/test"
#     folder="/inspire/hdd/project/qproject-fundationmodel/public/mhjiang/sii-cli-model/pr/data/pr_data_1800"
#     files=os.listdir(folder)
#     n=6
#     current=0
#     for file in files:
#         if "enhanced_data" in file and current<=n:
#             parts=file.replace("enhanced_data_","").replace(".jsonl","").split("_")
#             if len(parts)!=2:
#                 continue
#             repo_id=parts[0]
#             pr_number=parts[1]
#             print(f"处理 {repo_id} {pr_number}")
#             description = generate_task_description(repo_id, pr_number)
            
#             print("finished")
#             print("=="*20)
