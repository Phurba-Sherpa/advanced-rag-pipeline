import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (AgentMiddleware, AgentState,
                                         ModelRequest, dynamic_prompt)
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.messages import (HumanMessage, SystemMessage,
                                     convert_to_messages)
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
RETRIEVAL_K = 5
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vs = PGVector(embeddings=embed, collection_name="kb", connection=DATABASE_URL)
retriever = vs.as_retriever()
llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"}
    },
)


def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


def answer_question(question: str, history: list[dict] = []):
    """
    Answer the given question with RAG;
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    hist_msgs = convert_to_messages(history)
    messages = [
        SystemMessage(content=system_prompt),
        *hist_msgs,
        HumanMessage(content=question),
    ]
    response = llm.invoke(messages)
    return response.content, docs
