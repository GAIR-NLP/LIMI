#!/usr/bin/env python3
"""
A script to split a jsonl file containing GitHub PR data by repo_id and pr_number
"""

import json
import os
import argparse
from collections import defaultdict


def split_jsonl_file(input_file, output_dir=None):
    """
    Splits a jsonl file by repo_id and pr_number
    
    Args:
        input_file (str): Path to the input jsonl file
        output_dir (str, optional): Path to the output directory. If None, the directory of the input file will be used.
    """
    # Check if the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} does not exist")
    
    # Determine the output directory
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    # Create the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Reading file: {input_file}")
    
    # Get the file prefix (enhanced_data/raw_text/structured_data)
    base_filename = os.path.basename(input_file)
    if base_filename.startswith('enhanced_data_'):
        prefix = 'enhanced'
    elif base_filename.startswith('raw_text_'):
        prefix = 'raw'
    elif base_filename.startswith('structured_data_'):
        prefix = 'structured'
    else:
        # If the filename does not match the expected format, extract prefix from the filename
        prefix = base_filename.split('_')[0]
    
    # Used to count the number of lines for each combination
    data_groups = defaultdict(list)
    total_lines = 0
    
    # Read the file and group by repo_id and pr_number
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                repo_id = data.get('repo_id')
                pr_number = data.get('pr_number')
                
                if repo_id is None or pr_number is None:
                    print(f"Warning: Line {line_num} is missing 'repo_id' or 'pr_number' field")
                    continue
                
                key = (repo_id, pr_number)
                data_groups[key].append(line)
                total_lines += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: JSON parsing failed on line {line_num}: {e}")
                continue
    
    print(f"Successfully read {total_lines} lines, found {len(data_groups)} unique 'repo_id' and 'pr_number' combinations")
    
    # Write the grouped files
    created_count = 0
    for (repo_id, pr_number), lines in data_groups.items():
        output_filename = f"{prefix}_{repo_id}_{pr_number}.jsonl"
        output_path = os.path.join(output_dir, output_filename)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line + '\n')
        
        created_count += 1
        print(f"Created file: {output_filename} ({len(lines)} lines)")
    
    print(f"\nSplitting complete! A total of {created_count} files were created.")
    print(f"Output directory: {output_dir}")
    
    return created_count


def process_all_files():
    """Processes all files in the data directory"""
    base_data_dir = '/Users/yangxiao/projects/vscode/sii-cli-mid-training/data'
    
    target_files = [
        'enhanced_data_331293626.jsonl',
        'raw_text_331293626.jsonl',
        'structured_data_331293626.jsonl'
    ]
    
    total_created = 0
    for target_file in target_files:
        file_path = os.path.join(base_data_dir, target_file)
        if os.path.exists(file_path):
            print(f"\nProcessing file: {target_file}")
            print("=" * 50)
            try:
                created = split_jsonl_file(file_path, base_data_dir)
                total_created += created
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error processing file {target_file}: {e}")
        else:
            print(f"Warning: File {file_path} does not exist")
    
    print(f"\n\nTotal files created: {total_created}")
    return total_created


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Splits a jsonl file containing GitHub PR data by repo_id and pr_number')
    parser.add_argument('input_file', nargs='?', help='Path to the input jsonl file')
    parser.add_argument('-o', '--output', help='Path to the output directory (optional)')
    parser.add_argument('--all', action='store_true', help='Process all files in the data directory')
    
    args = parser.parse_args()
    
    if args.all or not args.input_file:
        # Process all relevant files in the data directory
        process_all_files()
    else:
        # Process a single file
        split_jsonl_file(args.input_file, args.output)


if __name__ == "__main__":
    main()