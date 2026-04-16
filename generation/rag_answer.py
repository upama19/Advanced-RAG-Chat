import os
from dotenv import load_dotenv
from openai import OpenAI
from chat.memory import ChatMemory
from routing.router import decide_tool

from retrieval.search import SemanticSearcher

# Load API key from .env
load_dotenv()
memory = ChatMemory()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def build_prompt(question, chunks, chat_history=""):
    """
    Build a RAG prompt using retieved chunks as context
    """
    context = "\n\n".join(
        f"[Source: {c['metadata']['source']} | Page: {c['metadata']['page_number']}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""
    You are a document-grounded research assistant.
    You ARE answering questions about the provided research paper.
    The provided context comes from the exact PDF the user is referring to.

    Rules:
    - Use ONLY the context below.
    - Do NOT say you cannot access the PDF.
    - If the answer is not in the context, say: "The paper does not explicitly mention this."    
    - If the question refers to experiments, results, or conclusions, focus on the corresponding sections.

    Conversation history:
    {chat_history}

    Context (extracted from the selected research paper):
    {context}

    Question:
    {question}

    Answer:
    """
    return prompt


def answer_question(question, active_document=None):

    tool = decide_tool(question)

    if tool == "direct" and active_document is None:
        prompt = f"Answer briefly:\n{question}"
        response = client.responses.create(model="gpt-4o-mini", input=prompt)
        return response.output_text, []

    # Load retriever
    searcher = SemanticSearcher(
        index_path="embeddings/faiss.index", metadata_path="embeddings/chunks.pkl"
    )

    memory.add_user_message(question)

    # Retrieve relevant chunks
    retrieved_chunks = searcher.search(
        question, top_k=5, active_document=active_document
    )

    # Build Prompt
    prompt = build_prompt(
        question, retrieved_chunks, chat_history=memory.get_formatted_history()
    )

    # Call OpenAI
    response = client.responses.create(model="gpt-4o-mini", input=prompt)

    answer_text = response.output_text
    memory.add_assistant_message(answer_text)

    return answer_text, retrieved_chunks


if __name__ == "__main__":
    print(answer_question("Hi")[0])
    print(answer_question("Explain U-Net", active_document="Unet.pdf")[0])
    print(answer_question("Summarize U-Net in two lines")[0])
