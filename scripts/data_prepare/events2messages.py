import json
import re
import os
import glob
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]

class ConversationConverter:    
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.tool_definitions = []
        self.system_prompt = ""
        
    def load_conversation_log(self) -> Dict[str, Any]:
        with open(self.input_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def normalize_parameter_types(self, obj):
        if isinstance(obj, dict):
            result = {}
            for key, value in obj.items():
                if key == "type" and isinstance(value, str):
                    type_mapping = {
                        "STRING": "string",
                        "ARRAY": "array", 
                        "OBJECT": "object",
                        "BOOLEAN": "boolean",
                        "INTEGER": "integer",
                        "NUMBER": "number"
                    }
                    result[key] = type_mapping.get(value, value.lower())
                else:
                    result[key] = self.normalize_parameter_types(value)
            return result
        elif isinstance(obj, list):
            return [self.normalize_parameter_types(item) for item in obj]
        else:
            return obj

    def convert_gemini_messages_to_openai(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        openai_messages = []
        
        for msg in messages:
            role = msg.get('role')
            parts = msg.get('parts', [])
            
            if role == 'user':
                for part in parts:
                    if 'text' in part:
                        openai_messages.append({
                            'role': 'user',
                            'content': part['text']
                        })
                    elif 'functionResponse' in part:
                        func_resp = part['functionResponse']
                        tool_content = func_resp.get('response', {}).get('output', '')
                        openai_messages.append({
                            'role': 'tool',
                            'content': tool_content,
                            'tool_call_id': func_resp.get('id', '')
                        })
                        
            elif role == 'model':
                text_content = ""
                tool_calls = []
                
                for part in parts:
                    if 'text' in part:
                        text_content += part['text']
                    elif 'functionCall' in part:
                        func_call = part['functionCall']
                        tool_calls.append({
                            'id': func_call.get('id', ''),
                            'type': 'function',
                            'function': {
                                'name': func_call.get('name', ''),
                                # 'arguments': json.dumps(func_call.get('args', {}), ensure_ascii=False)
                                'arguments': func_call.get('args', {})
                            }
                        })

                assistant_msg = {
                    'role': 'assistant',
                    'content': text_content
                }
                if tool_calls:
                    assistant_msg['tool_calls'] = tool_calls
                    
                openai_messages.append(assistant_msg)
        
        return openai_messages
    
    def extract_from_last_api_request(self, events: List[Dict[str, Any]]) -> tuple[str, List[ToolDefinition], List[Dict[str, Any]]]:
        last_api_request = None
        all_tools = []
        seen_tool_names = set()
        system_instruction = ""

        for event in events:
            if event.get('type') == 'api_request_full':
                data = event.get('data', {})
                full_request = data.get('full_request', {})

                for tool_group in full_request.get('tools', []):
                    if 'functionDeclarations' in tool_group:
                        for func_decl in tool_group['functionDeclarations']:
                            tool_name = func_decl.get('name', '')
                            if tool_name and tool_name not in seen_tool_names:
                                all_tools.append(ToolDefinition(
                                    name=tool_name,
                                    description=func_decl.get('description', ''),
                                    parameters=func_decl.get('parameters', {})
                                ))
                                seen_tool_names.add(tool_name)

                if 'system_instruction' in full_request:
                    system_instruction = full_request['system_instruction']

                last_api_request = event
        
        if not last_api_request:
            return "", [], []

        data = last_api_request.get('data', {})
        full_request = data.get('full_request', {})
        messages = full_request.get('messages', [])
        openai_messages = self.convert_gemini_messages_to_openai(messages)
        
        return system_instruction, all_tools, openai_messages
    
    def convert_to_openai_format(self, system_instruction: str, tool_definitions: List[ToolDefinition], messages: List[Dict[str, Any]]) -> Dict[str, Any]:

        tools = []
        for tool_def in tool_definitions:
            normalized_parameters = self.normalize_parameter_types(tool_def.parameters)
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": normalized_parameters
                }
            })

        enhanced_system_prompt = system_instruction
        
        if tool_definitions:
            tools_info = "\n\nYou have access to the following tools:\n\n"
            for i, tool_def in enumerate(tool_definitions, 1):
                tools_info += f"{i}. **{tool_def.name}**: {tool_def.description}\n"

                if tool_def.parameters and 'properties' in tool_def.parameters:
                    tools_info += "   Parameters:\n"
                    for param_name, param_info in tool_def.parameters['properties'].items():
                        param_type = param_info.get('type', 'unknown')
                        param_desc = param_info.get('description', '')
                        required_mark = " (required)" if param_name in tool_def.parameters.get('required', []) else ""
                        tools_info += f"   - {param_name} ({param_type}){required_mark}: {param_desc}\n"
                tools_info += "\n"
            
            tools_info += "Use these tools by making function calls in your responses when appropriate. Each tool call should include the function name and required parameters."
            enhanced_system_prompt += tools_info

        final_messages = []

        if enhanced_system_prompt:
            final_messages.append({
                "role": "system",
                "content": enhanced_system_prompt
            })

        final_messages.extend(messages)

        data_point = {
            "tools": tools,
            "messages": final_messages
        }
        
        return data_point
    
    def process(self) -> Dict[str, Any]:
        try:
            raw_data = self.load_conversation_log()

            events = raw_data.get('events', [])
            system_instruction, tool_definitions, messages = self.extract_from_last_api_request(events)
            
            training_data = self.convert_to_openai_format(system_instruction, tool_definitions, messages)
            
            result = {
                "source_file": self.input_file,
                "conversation_id": raw_data.get('id', ''),
                "model": raw_data.get('model', ''),
                "created": raw_data.get('created', ''),
                "message_count": len(messages),
                "tool_count": len(tool_definitions),
                "training_data": training_data
            }
            
            return result
            
        except Exception as e:
            print(f"Process {self.input_file} Error: {e}")
            import traceback
            traceback.print_exc()
            return None

class BatchConverter:    
    def __init__(self, raw_dir: str, output_dir: str):
        self.raw_dir = raw_dir
        self.output_dir = output_dir
        
    def find_readable_files(self) -> List[str]:
        pattern = os.path.join(self.raw_dir, "*.json")
        files = glob.glob(pattern)
        
        print(f"Find{len(files)} json files")
        return files
    
    def process_all_files(self):
        os.makedirs(self.output_dir, exist_ok=True)
        files = self.find_readable_files()
        
        if not files:
            print("There is no readable files.")
            return
        
        all_training_data = []
        processed_files = []
        failed_files = []
        
        for file_path in files:
            print(f"Process: {file_path}")
            
            converter = ConversationConverter(file_path)
            result = converter.process()
            
            if result and result.get('training_data'):
                all_training_data.append(result['training_data'])
                processed_files.append({
                    "file": file_path,
                    "conversation_id": result.get('conversation_id'),
                    "message_count": result.get('message_count'),
                    "tool_count": result.get('tool_count'),
                })
                print(f"  ✓ Process Success, contains {result.get('message_count')} messages, {result.get('tool_count')} tools")
            else:
                failed_files.append(file_path)
                print(f"  ✗ Process Failed.")

        output_file_jsonl = os.path.join(self.output_dir, "openai_training_data.jsonl")
        with open(output_file_jsonl, 'w', encoding='utf-8') as f:
            for item in all_training_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        output_file_json = os.path.join(self.output_dir, "openai_training_data.json")
        with open(output_file_json, 'w', encoding='utf-8') as f:
            json.dump(all_training_data, f, ensure_ascii=False, indent=2)

        report = {
            "total_files_found": len(files),
            "successfully_processed": len(processed_files),
            "failed_files": len(failed_files),
            "total_training_samples": len(all_training_data),
            "processed_files_details": processed_files,
            "failed_files_list": failed_files,
            "output_files": {
                "jsonl_format": output_file_jsonl,
                "json_format": output_file_json
            },
            "conversion_approach": "Extract from last API request (final correct approach)"
        }
        
        report_file = os.path.join(self.output_dir, "processing_report.json")
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"Files: {len(files)}")
        print(f"Process Success: {len(processed_files)}")
        print(f"Process Failed: {len(failed_files)}")
        print(f"Number: {len(all_training_data)}")
        print(f"JSONL output file: {output_file_jsonl}")
        print(f"JSON output file: {output_file_json}")
        print(f"Report: {report_file}")

def main():
    parser = argparse.ArgumentParser(description='Transfer to training data.')
    parser.add_argument('--raw_dir', '-r', required=True, help='Original data dir')
    parser.add_argument('--output_dir', '-o', required=True, help='Output data dir')
    
    args = parser.parse_args()
    
    print(f"Original data dir: {args.raw_dir}")
    print(f"Output data dir: {args.output_dir}")

    if not os.path.exists(args.raw_dir):
        print(f"Error: Not exist {args.raw_dir}")
        return 1

    batch_converter = BatchConverter(args.raw_dir, args.output_dir)

    batch_converter.process_all_files()
    
    print("Finish!")
    return 0

if __name__ == "__main__":
    exit(main())