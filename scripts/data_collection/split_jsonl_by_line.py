#!/usr/bin/env python3

import json
import os
from pathlib import Path
import glob
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

# threhold = 400
threhold = 2000
pr_num = 1


def load_valid_repo_ids(meta_file_path):
    valid_repo_ids = set()
    try:
        with open(meta_file_path, 'r', encoding='utf-8') as f:
            repo_data = json.load(f)
            for repo in repo_data:
                if 'repo_id' in repo:
                    valid_repo_ids.add(repo['repo_id'])
        print(f"Loaded {len(valid_repo_ids)} valid repo IDs from {meta_file_path}")
    except Exception as e:
        print(f"Error loading meta file: {e}")
        return None
    return valid_repo_ids

def get_pr_created_at_and_patch_length(data):
    created_at = None
    total_patch_length = 0
    has_md_files = False
    
    segments = data.get('segments', [])
    for segment in segments:
        if segment.get('segment_type') == 'pr_header':
            created_at = segment.get('created_at')

        if segment.get('segment_type') == 'pr_commit':
            files = segment.get('files', [])
            for file_info in files:
                filename = file_info.get('filename', '')
                if filename.lower().endswith('.md'):
                    has_md_files = True
                
                patch = file_info.get('patch')
                if patch is not None:
                    total_patch_length += len(patch)
    
    return created_at, total_patch_length, has_md_files

def process_jsonl_files():
    source_dir = Path("pr_data/pr_enhanced")
    target_dir = Path(f"pr_data/pr_data_{threhold}")
    meta_file = Path("pr_data/meta/repo_stats.json")
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载有效的repo_id列表
    valid_repo_ids = load_valid_repo_ids(meta_file)
    if valid_repo_ids is None:
        print("Failed to load valid repo IDs. Exiting.")
        return
    
    jsonl_files = list(source_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {source_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL file(s) to process")

    repo_prs = defaultdict(list)

    for jsonl_file in tqdm(jsonl_files):
        print(f"\nReading: {jsonl_file.name}")
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    data = json.loads(line)
                    repo_id = data.get('repo_id', None)
                    if repo_id in valid_repo_ids:
                        created_at, patch_length, has_md_files = get_pr_created_at_and_patch_length(data)

                        if created_at and not has_md_files:
                            repo_prs[repo_id].append({
                                'data': data,
                                'created_at': created_at,
                                'patch_length': patch_length,
                                'pr_number': data.get('pr_number')
                            })
                        
                except json.JSONDecodeError as e:
                    print(f"  Warning: Failed to parse line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"  Error processing line {line_num}: {e}")
                    continue

    total_saved = 0
    selected_summary = []
    
    print(f"\n\nFiltering PRs by creation time and patch length (max 2200), excluding .md files...")
    for repo_id, prs in repo_prs.items():
        if not prs:
            continue

        prs.sort(key=lambda x: x['created_at'])

        early_prs_count = max(len(prs) // 2, pr_num)
        early_prs = prs[:early_prs_count]

        early_prs_filtered = [pr for pr in early_prs if pr['patch_length'] <= threhold]

        if not early_prs_filtered:
            early_prs.sort(key=lambda x: x['patch_length'])
            selected_prs = early_prs[:pr_num]
            print(f"  Warning: Repo {repo_id} has no PRs with patch <= 2200, selecting smallest patches")
        else:
            early_prs_filtered.sort(key=lambda x: x['patch_length'], reverse=True)
            selected_prs = early_prs_filtered[:pr_num]
        
        for pr_info in selected_prs:
            data = pr_info['data']
            pr_number = pr_info['pr_number']
            output_filename = f"enhanced_data_{repo_id}_{pr_number}.jsonl"
            output_path = target_dir / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(data, out_f, ensure_ascii=False)
                out_f.write('\n')
            
            total_saved += 1
            selected_summary.append({
                'repo_id': repo_id,
                'pr_number': pr_number,
                'created_at': pr_info['created_at'],
                'patch_length': pr_info['patch_length']
            })

    print(f"\n=== Summary ===")
    print(f"Total repos processed: {len(repo_prs)}")
    print(f"Total PRs saved: {total_saved}")
    print(f"Output directory: {target_dir.absolute()}")

    print(f"\nSample of selected PRs (showing first 10):")
    for pr in selected_summary[:10]:
        print(f"  Repo {pr['repo_id']}, PR #{pr['pr_number']}: "
              f"created at {pr['created_at'][:10]}, patch length: {pr['patch_length']}")

if __name__ == "__main__":
    process_jsonl_files()