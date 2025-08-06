# 🚀 ScoutOS

[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Note:** 🚧 This project is currently under active development. We're continuously improving the codebase and adding new features. Future updates will include enhanced file/folder structure and additional capabilities.

## 🌟 About

ScoutOS is a personal AI assistant project designed to provide hands-on experience with modern AI/ML technologies. This project serves as a practical learning platform for working with:

- Large Language Models (LLMs)
- Vector search and embeddings
- Semantic search and memory
- Natural Language Processing (NLP)
- Web integration and API development
- Modern Python development practices

## 🛠️ Features

- **Conversational AI**: Powered by Mistral LLM for natural language understanding
- **Semantic Search**: Advanced document indexing and retrieval using vector embeddings
- **Web Integration**: Real-time web search capabilities
- **Memory System**: Short-term and long-term memory for context-aware responses
- **Modular Architecture**: Designed for extensibility and maintainability

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Ollama (for local LLM inference)
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/GarlonJG/ScoutOS.git
   cd ScoutOS
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running ScoutOS

1. Start the Ollama server (if using local LLM):
   ```bash
   ollama serve
   ```

2. In a new terminal, run the application:
   ```bash
   streamlit run scoutos_starter.py
   ```

3. Open your browser to the provided local URL (typically http://localhost:8501)

## 🧠 Learning Objectives

This project was created to gain hands-on experience with:

- **Python Development**: Modern Python practices, async/await, and package management
- **LLM Integration**: Working with local and cloud-based language models
- **Vector Databases**: Implementing semantic search with FAISS and embeddings
- **Web Development**: Building interactive UIs with Streamlit
- **AI/ML Concepts**: Fine-tuning, prompt engineering, and model optimization
- **Software Architecture**: Designing modular, maintainable systems

## 📂 Project Structure

```
scoutos/
├── data/                   # Sample documents and data
├── utils/                  # Utility functions and helpers
│   ├── llm.py             # LLM interface and utilities
│   ├── memory.py          # Memory management
│   └── web_search.py      # Web search functionality
├── semantic_document.py   # Document processing and search
├── simple_vector_search.py # Vector search implementation
└── scoutos_starter.py     # Main application entry point
```

*Note: The project structure is being improved in upcoming versions for better organization and scalability.*

## 🤝 Contributing

Contributions are welcome! While this is primarily a personal learning project, I'm open to suggestions and improvements. Please feel free to submit issues or pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The open-source community for their invaluable tools and libraries
- Mistral AI for their powerful open-weight models
- The Streamlit team for making web UIs accessible to Python developers

---

Built with ❤️ by [Garlon](https://github.com/GarlonJG)
