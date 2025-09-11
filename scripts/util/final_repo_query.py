#!/usr/bin/env python3


import json
import argparse
import sys
import time


def query_repo_stream(json_file, target_repo_id, verbose=False):

    processed = 0
    start_time = time.time()
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                processed += 1
                
                if verbose and processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Process {processed:,} records, take {elapsed:.2f} seconds")
                
                if str(item.get('repo_id', '')) == str(target_repo_id):
                    if verbose:
                        elapsed = time.time() - start_time
                        print(f"Find Fitness! Process {processed:,} Records, take {elapsed:.2f} seconds")
                    return item.get('repo_name')
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Not Find{target_repo_id}, Process{processed:,} records in total, take{elapsed:.2f} seconds")
            
            return None
            
    except FileNotFoundError:
        print(f"file not exist: {json_file}")
        return None
    except json.JSONDecodeError as e:
        print(f"JSON Decode Error: {e}")
        return None
    except Exception as e:
        print(f"Query Error: {e}")
        return None


def batch_query_repos(json_file, repo_ids, verbose=False):
    """Batch query"""
    target_set = {str(rid) for rid in repo_ids}
    results = {str(rid): None for rid in repo_ids}
    found = 0
    processed = 0
    start_time = time.time()
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for item in data:
                processed += 1
                
                if verbose and processed % 10000 == 0:
                    elapsed = time.time() - start_time
                    print(f"Processed {processed:,} records, found {found}/{len(repo_ids)}, elapsed {elapsed:.2f} seconds")
                
                repo_id = str(item.get('repo_id', ''))
                if repo_id in target_set:
                    results[repo_id] = item.get('repo_name')
                    found += 1
                    
                    if verbose:
                        print(f"Found: {repo_id} -> {item.get('repo_name')}")
                    
                    if found == len(repo_ids):
                        if verbose:
                            elapsed = time.time() - start_time
                            print(f"All targets found! Processed {processed:,} records, elapsed {elapsed:.2f} seconds")
                        break
            
            if verbose:
                elapsed = time.time() - start_time
                print(f"Batch query completed! Found {found}/{len(repo_ids)}, total processed {processed:,} records, elapsed {elapsed:.2f} seconds")
            
            return results
            
    except Exception as e:
        print(f"Batch query error: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Query repo_name by repo_id')
    parser.add_argument('json_file', help='JSON file path')
    parser.add_argument('-i', '--id', help='repo_id to query')
    parser.add_argument('-b', '--batch', help='Batch query file (one repo_id per line)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    parser.add_argument('-o', '--output', help='Save results to file')
    
    args = parser.parse_args()
    
    if not args.id and not args.batch:
        parser.error("Must specify either --id or --batch")
    
    if args.id and args.batch:
        parser.error("--id and --batch cannot be used together")
    
    # Single query
    if args.id:
        result = query_repo_stream(args.json_file, args.id, args.verbose)
        if result:
            print(f"repo_id: {args.id}")
            print(f"repo_name: {result}")
        else:
            print(f"repo_id not found: {args.id}")
            sys.exit(1)
    
    # Batch query
    elif args.batch:
        try:
            with open(args.batch, 'r') as f:
                repo_ids = [line.strip() for line in f if line.strip()]
            
            print(f"Starting batch query for {len(repo_ids)} repo_ids...")
            results = batch_query_repos(args.json_file, repo_ids, args.verbose)
            
            output_lines = []
            found_count = 0
            
            for repo_id, repo_name in results.items():
                if repo_name:
                    line = f"{repo_id}\t{repo_name}"
                    found_count += 1
                else:
                    line = f"{repo_id}\t[NOT FOUND]"
                
                output_lines.append(line)
                print(line)
            
            print(f"\nBatch query completed: found {found_count}/{len(repo_ids)}")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))
                print(f"Results saved to: {args.output}")
                
        except FileNotFoundError:
            print(f"Batch file not found: {args.batch}")
            sys.exit(1)
        except Exception as e:
            print(f"Batch query failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    # Interactive mode
    if len(sys.argv) == 1:
        json_file = "./util/star_5_final.json"
        print("=== Repo Query Tool ===")
        print(f"Using file: {json_file}")
        print("Enter 'quit' to exit")
        
        while True:
            try:
                repo_id = input("\nPlease enter repo_id: ").strip()
                if repo_id.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not repo_id:
                    continue
                
                result = query_repo_stream(json_file, repo_id, verbose=True)
                if result:
                    print(f"✓ Found: {result}")
                else:
                    print("✗ Not found")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        main()
