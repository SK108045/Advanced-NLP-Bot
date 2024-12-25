# Advanced-NLP-Bot
An AI bot powered by a Retrieval Augmented Generation (RAG) pipeline that combines document intelligence with conversational AI. Built using Ollama models and advanced embedding techniques for semantic search and contextual understanding.
## Key Features

- **Intelligent RAG Pipeline**: Uses semantic embeddings for precise information retrieval from documents
- **Dual Interface System**: 
  - Document Analysis Assistant with PDF/TXT support
  - Conversational AI with voice capabilities
- **Advanced NLP Features**:
  - Real-time text embedding using nomic-embed-text
  - Semantic similarity search for context retrieval
  - Streaming response generation
- **Voice Integration**:
  - Text-to-speech API by Deepgram's `Aura Stella model` with Cached audio responses for improved performance

## Technical Architecture

- **Embedding Engine**: Uses **`nomic-embed-text`** for document and query vectorization
- **LLM Backend**: The project uses Ollama with **`llama:3.2:1b`** model
- **Document Processing**: PyPDF2 for PDF parsing with enhanced text preprocessing
- **Frontend**: Streamlit for interactive UI components
- **Audio System**: Deepgram API for high-quality voice synthesis
  
## Project Setup Guide

### 1. Environment Setup

1. Install Python 3.8+ and Git
2. Install Ollama from [here]([https://ollama.ai](https://ollama.com/download)) and pull `llama3.2:1b` and `nomic-embed-text` models
3. Clone this repository:
```bash
git clone https://github.com/SK108045/Advanced-NLP-Bot.git
```
4. Create a virtual environment and install the requirements . Also add your Deepgram API to the `.env` file
   
