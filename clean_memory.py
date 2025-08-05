import json

MEMORY_FILE = "chat_memory.json"

def clean_memory():
    with open(MEMORY_FILE, "r") as f:
        memory = json.load(f)

    cleaned_memory = [m for m in memory if m.get("assistant") != "thinking..."]

    removed_count = len(memory) - len(cleaned_memory)
    with open(MEMORY_FILE, "w") as f:
        json.dump(cleaned_memory, f, indent=2)

    print(f"Removed {removed_count} incomplete entries.")

if __name__ == "__main__":
    clean_memory()