#!/usr/bin/env python3
"""
Simplified repo query script
"""

import json
import sys

def simple_query(repo_id):
    """Simple query function"""
    try:
        with open('star_5_final.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            if str(item.get('repo_id')) == str(repo_id):
                return item.get('repo_name')
        
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        repo_id = sys.argv[1]
        result = simple_query(repo_id)
        if result:
            print(f"repo_id: {repo_id}")
            print(f"repo_name: {result}")
        else:
            print(f"Repo ID not found: {repo_id}")
    else:
        print("Usage: python3 simple_query.py <repo_id>")