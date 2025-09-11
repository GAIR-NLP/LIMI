import os
import json
import pdb
import argparse
import pandas as pd
from copy import deepcopy
from transformers import AutoTokenizer

def main():

    parser = argparse.ArgumentParser(description='Transfer JSON data to Parquet')
    parser.add_argument('--tokenizer_path', '-t', required=True, help='Tokenizer Model Path')
    parser.add_argument('--input_json', '-i', required=True, help='OpenAI JSON File Path')
    parser.add_argument('--output_parquet', '-o', required=True, help='Output Parquet File')
    parser.add_argument('--max_tokens', '-m', type=int, default=128000, help='Max token (Default: 128000)')
    parser.add_argument('--duplicate_times', '-d', type=int, default=10, help='Duplicate Times(Default: 10)')
    
    args = parser.parse_args()
    
    print(f"Tokenizer Path: {args.tokenizer_path}")
    print(f"Input JSON File: {args.input_json}")
    print(f"Output Parquet Files: {args.output_parquet}")
    print(f"Max token: {args.max_tokens}")
    print(f"duplicate_times: {args.duplicate_times}")

    if not os.path.exists(args.input_json):
        print(f"Error: no {args.input_json}")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    print(tokenizer.chat_template)

    with open(args.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(len(data))

    def fix_tool_calls(messages):
        for m in messages:
            if m.get("role") == "assistant" and "tool_calls" in m:
                for tc in m["tool_calls"]:
                    func = tc.get("function", {})
                    args = func.get("arguments")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except Exception:
                            func["arguments"] = {}
            elif m.get("role") == "tool" and isinstance(m.get("content"), str):
                pass
        return messages

    save_data = []
    key_type = {}
    for sample in data:
        conflict = False
        full_prompt = tokenizer.apply_chat_template(
            fix_tool_calls(sample["messages"]),
            tools=sample.get("tools", []),
            tokenize=False,
            add_generation_prompt=False
        )
        for idx in range(len(sample['messages'])):
            if "tool_calls" in sample["messages"][idx]:
                for call in sample["messages"][idx]["tool_calls"]:
                    call_new = deepcopy(call)
                    for key in call["function"]["arguments"].keys():
                        if key in key_type.keys():
                            if type(call["function"]["arguments"][key]) != key_type[key]:
                                print(f"{key} | {type(call['function']['arguments'][key])} v.s. key_type={key_type[key]}")
                                # call_new["function"]["arguments"].pop(key)
                                conflict = True
                        else:
                            key_type[key] = type(call["function"]["arguments"][key])
                    # sample["messages"][idx]["tool_calls"] = call_new
                # sample["messages"][idx]["tool_calls"] = json.dumps(sample["messages"][idx]["tool_calls"])
        # if len(tokenizer.encode(full_prompt, add_special_tokens=False)) > 128000 or conflict:
        if len(tokenizer.encode(full_prompt, add_special_tokens=False)) > args.max_tokens or conflict:
            continue
        save_data.append(sample)


    save_data = save_data * args.duplicate_times
    print(len(save_data))

    output_dir = os.path.dirname(args.output_parquet)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    pd.DataFrame(save_data).to_parquet(args.output_parquet)
    
    return 0

if __name__ == "__main__":
    exit(main())