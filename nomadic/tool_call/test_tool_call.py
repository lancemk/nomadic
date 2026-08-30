import os
import json
import re
# import mlx_vlm  # <- Switch to mlx_vlm to support gemma4_unified
import sys
import mlx_vlm

# ==========================================
# 1. DEFINE OS FILESYSTEM OPERATIONS
# ==========================================
def write_local_file(path, content):
    try:
        target_path = os.path.abspath(path)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "message": f"Successfully wrote file to {target_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def read_local_file(path):
    try:
        target_path = os.path.abspath(path)
        with open(target_path, 'r', encoding='utf-8') as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ==========================================
# 2. DEFINITIONS USING THE GEMMA 4 '<|\"|>' STRUCTURAL TOKENS
# ==========================================
WRITE_FILE_SCHEMA = (
    "<|tool>{\n"
    '  "name": <|\\"|>write_local_file<|\\"|>,\n'
    '  "description": <|\\"|>Write text content to a local file path.<|\\"|>,\n'
    '  "parameters": {\n'
    '    "type": <|\\"|>object<|\\"|>,\n'
    '    "properties": {\n'
    '      "path": {"type": <|\\"|>string<|\\"|>, "description": <|\\"|>Local target path<|\\"|>},\n'
    '      "content": {"type": <|\\"|>string<|\\"|>, "description": <|\\"|>Text content to save<|\\"|>}\n'
    "    },\n"
    '    "required": [<|\\"|>path<|\\"|>, <|\\"|>content<|\\"|>]\n'
    "  }\n"
    "}<|tool>"
)

READ_FILE_SCHEMA = (
    "<|tool>{\n"
    '  "name": <|\\"|>read_local_file<|\\"|>,\n'
    '  "description": <|\\"|>Read text content from a local file path.<|\\"|>,\n'
    '  "parameters": {\n'
    '    "type": <|\\"|>object<|\\"|>,\n'
    '    "properties": {\n'
    '      "path": {"type": <|\\"|>string<|\\"|>, "description": <|\\"|>Path of target file to open<|\\"|>}\n'
    "    },\n"
    '    "required": [<|\\"|>path<|\\"|>]\n'
    "  }\n"
    "}<|tool>"
)

# ==========================================
# 3. AGENTIC COMPILATION LOOP
# ==========================================
def run_agentic_loop(model, processor, user_intention):
    print(f"\n[User Intention]: {user_intention}")
    
    system_prompt = (
        "<|turn>system\n"
        "You are an automated OS file administrator. You have immediate access to filesystem tools.\n"
        f"{WRITE_FILE_SCHEMA}\n"
        f"{READ_FILE_SCHEMA}\n"
        "Think carefully inside <|think|> tags before electing a tool block.\n"
        "If you want to use a tool, output a single call structure explicitly like:\n"
        "sub_tool:function_name{arg:<|\"|>val<|\"|>}\n"
        "Never invent parameters outside your provided specs.<turn|>\n"
    )
    
    user_turn = f"<|turn>user\n{user_intention}<turn|>\n"
    prompt = f"{system_prompt}{user_turn}<|turn>model\n"
    
    # Gemma 4 core structural stops
    stop_sequences = ["<turn|>", "<|tool_response>"]
    
    print("[Engine Executing Local Weights Inference via unified VLM backend...]")
    
    # Generate calls using mlx_vlm's orchestration wrapper
    generation_output = mlx_vlm.generate(
        model, 
        processor, 
        prompt=prompt, 
        max_tokens=2048, 
        stop_words=stop_sequences,
        verbose=False
    )
    
    # Extract the text property from the GenerationResult wrapper safely
    # Robust Type Verification:
    if generation_output is None:
        generated_text = ""
    elif hasattr(generation_output, "text"):
        generated_text = str(generation_output.text)
    else:
        generated_text = str(generation_output)

    print(f"\n--- Model Response Block ---\n{generated_text}\n---------------------------")
    
    tool_call_match = re.search(r"<\|tool_call\|>call:(\w+)\{(.*?)\}", generated_text)
    
    if tool_call_match:
        function_name = tool_call_match.group(1)
        raw_arguments = tool_call_match.group(2)
        
        clean_args = {}
        arg_pairs = re.findall(r"(\w+):\s*<|\\\"|>(.*?)<|\\\"|>", raw_arguments)
        for key, value in arg_pairs:
            clean_args[key] = value
            
        print(f"\n[CLI Action Intercepted]: Invoking Call -> Function: '{function_name}' with Args: {clean_args}")
        
        execution_result = {}
        if function_name == "write_local_file":
            execution_result = write_local_file(clean_args.get('path'), clean_args.get('content'))
        elif function_name == "read_local_file":
            execution_result = read_local_file(clean_args.get('path'))
            
        print(f"[OS Execution Output]: {execution_result}")
        
        clean_json_res = json.dumps(execution_result).replace('"', '<|\\"|>')
        tool_feedback = f"<|tool_response>response:{function_name}{clean_json_res}<turn|>\n"
        
        final_prompt = f"{prompt}{generated_text}{tool_feedback}<|turn>model\n"
        
        final_output = mlx_vlm.generate(
            model, 
            processor, 
            prompt=final_prompt, 
            max_tokens=512, 
            stop_words=["<turn|>"],
            verbose=False
        )
        
        # Robust Type Verification:
        if final_output is None:
            final_response = "Error: Engine returned an empty response graph."
        elif hasattr(final_output, "text"):
            final_response = str(final_output.text)
        else:
            final_response = str(final_output)

        print(f"\n[Final Agent Message to User]:\n{final_response.strip()}")
    else:
        print("\n[Final Agent Message to User (No Tool Required)]:")
        clean_msg = re.sub(r"<|think|>.*?<\|think\|>", "", generated_text, flags=re.DOTALL)
        print(clean_msg.strip())

# ==========================================
# 4. RUN EXECUTION PIPELINE
# ==========================================
if __name__ == "__main__":
    # model_id = "mlx-community/gemma-4-12b-bf16"
    model_id = "mlx-community/gemma-4-26b-a4b-it-4bit"

    print(f"Loading unified architecture [{model_id}] using Apple Silicon optimized hardware mappings...")
    # Load model and processor components using mlx_vlm
    model, processor = mlx_vlm.load(model_id)
    
    test_prompt = "Create a summary layout file named z_myname.txt containing the exact message: 'System status green.'"
    run_agentic_loop(model, processor, test_prompt)