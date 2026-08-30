import os
import json
from PIL import Image
import mlx_vlm
from mlx_lm.utils import generate

# --- Step 1: Define Actual OS-Level Functions ---
def read_local_file(path):
    """Safely reads a text file from the filesystem."""
    try:
        # Prevent accessing system files outside a designated workspace if desired
        with open(path, 'r', encoding='utf-8') as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def write_local_file(path, content):
    """Writes content to the local filesystem."""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "message": f"Successfully wrote to {path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    

# --- Step 2: Define JSON Tool Schemas for the Model's Context ---
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "write_local_file",
            "description": "Write text content to a local file path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Local file path"},
                    "content": {"type": "string", "description": "Text content to write"}
                },
                "required": ["path", "content"]
            }
        }
    }
]

# --- Step 3: Initialize Model & Prepare the Multimodal Prompt ---
# Using Qwen2-VL via mlx_vlm
model_path = "mlx-community/Qwen2-VL-7B-Instruct-4bit"
model, processor = mlx_vlm.load(model_path)

# System prompt instructs the model on how to format its thoughts for tools
system_instruction = (
    "You are an OS automation tool. You have access to local file system tools. "
    f"Available tools: {json.dumps(tools_schema)}. "
    "If you need to use a tool, you MUST respond in pure JSON format matching this schema: "
    '{"tool_call": {"name": "function_name", "arguments": {...}}}'
)

user_prompt = "Transcribe the text found within this image and save it to a file named 'extracted_notes.txt'."
image_path = "user_input_note.png" 
image = Image.open(image_path)

# Apply Chat Template via the processor
messages = [
    {"role": "system", "content": system_instruction},
    {"role": "user", "content": f"<|vision_start|><|image_pad|><|vision_end|>{user_prompt}"}
]

# Process both inputs for MLX
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = mlx_vlm.generate(model, processor, prompt, image, verbose=False)

# --- Step 4: Intercept Output and Execute Tool Call ---
print(f"Model Output Raw: {output}")

try:
    # Attempt to parse if the model outputted a structured tool call
    parsed_output = json.loads(output.strip())
    if "tool_call" in parsed_output:
        tool_name = parsed_output["tool_call"]["name"]
        args = parsed_output["tool_call"]["arguments"]
        
        if tool_name == "write_local_file":
            # Execute the local filesystem action
            result = write_local_file(args['path'], args['content'])
            print(f"\n[CLI Action Executed]: {result}")
            
except json.JSONDecodeError:
    # If the model just replied with text instead of a tool call
    print(f"\n[Model Response]: {output}")