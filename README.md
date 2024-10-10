# Assalim's LLM Project

This project is an implementation of a **custom Large Language Model (LLM)** that processes and queries documents stored in a **VectorStore** and, if necessary, searches for answers on the internet. The system is designed to optimize information retrieval and provide fast and accurate responses.

## Features

- **File Upload**: Upload documents in different formats (PDF, DOCX, etc.) and save them in the vector database using embeddings.
- **LLM Queries**: Ask anything. If the answer is in the context of the stored documents, the LLM will respond based on that content. Otherwise, the system will search online to find the best answer.
- **Vector Processing**: Uses embeddings to perform efficient searches within loaded documents.
- **Groq Integration**: The system uses a Groq LLM model to answer questions based on documents or the internet.

## Technologies Used

- **LangChain**: For loading and processing different types of files.
- **Tavily, Groq, Chroma**: To perform fast and accurate vector searches.
- **Embeddings with Transformers**: Used to generate embeddings from documents and store them in the vector database.

## How to Use

### Requirements

- Python 3.8+
- Install the required dependencies:

```bash
pip install -r requirements.txt
```

# Running the System

## Starting the Script: To start, just run the main script:

```bash
python llm_chat.py
```

## Add Files to VectorStore:

### The system will prompt you to choose one of the options:

  - Press 1 to provide the full path of a file from your computer and add it to the VectorStore.
  - Press 2 to start asking questions to the LLM.
    
### Interacting with the LLM:

  - If the answer to your question is within the context of the loaded documents, the LLM will provide that response.
  - If the answer is not present, the system will perform an internet search and provide a detailed summary.
  - To end the chat, type exit.

## Estrutura do Projeto

```bash
├── services/
│   ├── loader.py        # File loading
│   ├── groq_model.py    # Groq model for LLM queries
│   ├── embed.py         # Embeddings and vector storage
├── main.py              # Main script to run the system
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```
