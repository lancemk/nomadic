import os

# ==========================================
# 1. DEFINE CORE OS FILESYSTEM OPERATIONS
# ==========================================
def write_local_file(path, content):
    """Executes file writing inside the host operating system environment."""
    try:
        # Resolve path relative to current execution context
        target_path = os.path.abspath(path)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return {"status": "success", "message": f"Successfully wrote file to {target_path}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

def read_local_file(path):
    """Executes file reading inside the host operating system environment."""
    try:
        target_path = os.path.abspath(path)
        with open(target_path, 'r', encoding='utf-8') as f:
            return {"status": "success", "content": f.read()}
    except Exception as e:
        return {"status": "error", "message": str(e)}