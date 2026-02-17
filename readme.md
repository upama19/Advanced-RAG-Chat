# PDF Chat using RAG

This project is a simple Retrieval-Augmented Generation (RAG) application that allows a user to upload a PDF research paper and ask questions about it through a chat interface. The answers are generated using an LLM and are grounded only in the uploaded document.

## What this project does

- Upload a PDF file through the UI
- Automatically process the PDF (chunking and embeddings)
- Store embeddings in a FAISS vector index
- Retrieve relevant content based on the user’s question
- Generate answers using an LLM with retrieved context
- Provide a chat-style interface using Streamlit

## Tech used

- Python
- Streamlit
- FAISS
- Sentence Transformers
- OpenAI API

## How to run

1. Clone the repository
```
git clone <your-repo-url>
cd advanced-rag-chat
```

2. Create and activate a virtual environment
```
python -m venv rag-chat
rag-chat\Scripts\activate
```

3. Install dependencies
```
pip install -r requirements.txt
```

4. Add your OpenAI API key  
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_api_key_here
```

5. Run the application
```
streamlit run ui/app.py
```

## How to use

1. Open the app in the browser
2. Upload a PDF document
3. Wait for the document to process
4. Ask questions related to the uploaded paper

## Notes

- The system answers questions only based on the uploaded PDF
- If the answer is not present in the document, the model will say so
- Only one document is active at a time

## Future improvements

- Support multiple PDFs
- Show source references
- Improve chunking strategy
- Deploy online
