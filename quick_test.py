import os, json, requests, yaml
import time

payload = {
    "model": "mistral",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke about llamas"}
    ],
    "stream": False,
    "options": {"temperature": 0.7}
}

start = time.time()
r = requests.post("http://localhost:11434/api/chat", json=payload)
print("Time:", time.time() - start)
print("Response:", r.json())