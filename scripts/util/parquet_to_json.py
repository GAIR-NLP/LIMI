#!/usr/bin/env python3
"""
A script to convert a parquet file to JSON format
"""

import pandas as pd
import json
import argparse
import os
from pathlib import Path


def parquet_to_json(parquet_file, output_file=None, orient='records', indent=2):
    """
    Converts a parquet file to a JSON file
    
    Args:
        parquet_file (str): Path to the input parquet file
        output_file (str, optional): Path to the output JSON file. If None, the input filename will be used.
        orient (str): JSON format orientation. Options: 'records', 'index', 'values', 'split', 'table'
        indent (int): Number of spaces for JSON indentation
    """
    try:
        # Check if the input file exists
        if not os.path.exists(parquet_file):
            raise FileNotFoundError(f"File {parquet_file} does not exist")
        
        print(f"Reading parquet file: {parquet_file}")
        
        # Read the parquet file
        df = pd.read_parquet(parquet_file)
        
        print(f"Successfully read data, with {len(df)} rows and {len(df.columns)} columns")
        print(f"Column names: {list(df.columns)}")
        
        # If output file is not specified, use the input filename
        if output_file is None:
            base_name = Path(parquet_file).stem
            output_file = f"{base_name}.json"
        
        print(f"Converting to JSON format...")
        
        # Convert to JSON
        json_data = df.to_json(orient=orient, indent=indent, force_ascii=False)
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_data)
        
        print(f"Conversion complete! Output file: {output_file}")
        print(f"File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
        
        return output_file
        
    except Exception as e:
        print(f"Conversion failed: {str(e)}")
        raise


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert a parquet file to JSON format')
    parser.add_argument('input_file', help='Path to the input parquet file')
    parser.add_argument('-o', '--output', help='Path to the output JSON file (optional)')
    parser.add_argument('--orient', default='records', 
                       choices=['records', 'index', 'values', 'split', 'table'],
                       help='JSON format orientation (default: records)')
    parser.add_argument('--indent', type=int, default=2, help='Number of spaces for JSON indentation (default: 2)')
    parser.add_argument('--preview', action='store_true', help='Preview the first 5 rows of data')
    
    args = parser.parse_args()
    
    # If the preview option is specified
    if args.preview:
        try:
            df = pd.read_parquet(args.input_file)
            print("Data preview (first 5 rows):")
            print(df.head().to_string())
            print(f"\nData info:")
            print(f"Rows: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            print(f"Column names: {list(df.columns)}")
            return
        except Exception as e:
            print(f"Preview failed: {str(e)}")
            return
    
    # Execute the conversion
    parquet_to_json(args.input_file, args.output, args.orient, args.indent)


if __name__ == "__main__":
    # If no command-line arguments are provided, convert star_5_final.parquet in the current directory
    import sys
    if len(sys.argv) == 1:
        parquet_file = "star_5_final.parquet"
        if os.path.exists(parquet_file):
            print("No command-line arguments provided. Converting star_5_final.parquet in the current directory.")
            parquet_to_json(parquet_file)
        else:
            print("Usage:")
            print("python parquet_to_json.py <parquet_file_path> [options]")
            print("Or place star_5_final.parquet in the current directory and run directly.")
    else:
        main()