# Patient Data Extraction and RAG System

This project consists of two main components: a Patient Data Extraction tool and a Retrieval-Augmented Generation (RAG) system. Both components are designed to process and analyze medical and general information efficiently.

## 1. Patient Data Extraction

This tool extracts structured patient data from unstructured text using OpenAI's language model and Pydantic for data validation.

### Features

- Extracts patient information including name, gender, age, weight, height, BMI, and chief medical complaint
- Uses Pydantic for robust data schema definition and validation

### Installation

```bash
pip install -r requirements.txt
```

### Usage

There are two files for patient data extraction:

1. Using LLama 3.2 locally with Ollama (may not give consistent structured output):
```bash
python patientdata_ollama.py
```

2. Using OpenAI (provides mostly consistent structured output):
```bash
python patientdata_openai.py
```

**Note:** For the OpenAI version, add your OPENAI_KEY to the code before running.

## 2. RAG System

This system retrieves semantically similar documents from locally stored Chroma DB vector embeddings of Wikipedia pages and answers user queries using a local LLama 3.2 3B model.

### Features

- Uses BAAI/bge-large-en-v1.5 embedding model from HuggingFace
- Employs Ollama to run LLama 3.2 3B model locally
- Retrieves relevant Wikipedia documents based on semantic similarity

### Ollama Service Management

Start Ollama:
```bash
sudo systemctl start ollama
```

Check Ollama status:
```bash
sudo systemctl status ollama
```

Stop Ollama:
```bash
sudo systemctl stop ollama
```

### Ollama Model Management

Run Llama 3.2:
```bash
ollama run llama3.2
```

List all models:
```bash
ollama ls
```

Remove a model:
```bash
ollama rm <model_name>
```

### Usage

Run the RAG system with:

```bash
python rag_app.py
```

Enter your query in the terminal.