#!/bin/bash

# Load environment variables from .env (optional)
if [ -f .env ]; then
  echo "🔧 Loading environment variables from .env..."
  set -o allexport
  source .env
  set +o allexport
fi

# Fallback defaults if not set
: "${OLLAMA_NUM_THREAD:=2}"
: "${OLLAMA_NUM_CPU:=4}"

echo "🧠 Using OLLAMA_NUM_THREAD=$OLLAMA_NUM_THREAD"
echo "🧠 Using OLLAMA_NUM_CPU=$OLLAMA_NUM_CPU"

# Launch Ollama in background
echo "🚀 Starting Ollama server..."
OLLAMA_NUM_THREAD=$OLLAMA_NUM_THREAD \
OLLAMA_NUM_CPU=$OLLAMA_NUM_CPU \
ollama serve > ollama.log 2>&1 &

OLLAMA_PID=$!
echo "📡 Ollama server running with PID $OLLAMA_PID"

# Wait briefly to make sure it's started
sleep 2

# Start Streamlit app
echo "🎨 Launching Streamlit app..."
streamlit run scoutos_starter.py

# Cleanup Ollama process on exit
echo "🛑 Shutting down Ollama (PID $OLLAMA_PID)..."
kill $OLLAMA_PID