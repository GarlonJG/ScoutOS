def search_memory(memory_key, query):
    return f"(memory context for {memory_key})"

def store_memory(memory_key, query, response):
    print(f"[Memory saved for {memory_key}] {query} => {response[:60]}")
