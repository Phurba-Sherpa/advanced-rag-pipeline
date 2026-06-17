import json
import os
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
RETRIEVAL_K = 20
TOP_K = 10
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""
_retry_wait = wait_exponential(multiplier=1, min=2, max=30)
_retry_stop = stop_after_attempt(3)

embed = OllamaEmbeddings(model="nomic-embed-text", base_url="http://localhost:11434")
vs = PGVector(embeddings=embed, collection_name="kb", connection=DATABASE_URL)
retriever = vs.as_retriever()


class RankOrder(BaseModel):
    order: list[int] = Field(
        description="The order of relevance of chunks, from most relevant to least relevant, by chunk id number"
    )


_ollama_kwargs = {
    "model": "gpt-oss:120b",
    "base_url": "https://ollama.com",
    "client_kwargs": {
        "headers": {"Authorization": f"Bearer {os.environ.get('OLLAMA_API_KEY')}"}
    },
}

# Plain LLM for text generation (rewrite_query, answer_question)
llm = ChatOllama(**_ollama_kwargs)

# Rerank LLM — same plain LLM; we parse the plain-text ranked list response manually
llm_rerank = ChatOllama(**_ollama_kwargs)


def _parse_rank_order(text: str, max_id: int) -> list[int]:
    """Parse a plain-text ranked list like '1, 3, 2' or '{"order": [1,3,2]}' into a list of ints."""
    # Try JSON first
    try:
        import json as _json

        obj = _json.loads(text)
        if isinstance(obj, dict) and "order" in obj:
            order = [int(x) for x in obj["order"]]
        elif isinstance(obj, list):
            order = [int(x) for x in obj]
        else:
            raise ValueError("unexpected JSON shape")
        return [i for i in order if 1 <= i <= max_id]
    except Exception:
        pass
    # Fall back to extracting all integers from plain text
    nums = [int(m) for m in re.findall(r"\d+", text)]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for n in nums:
        if 1 <= n <= max_id and n not in seen:
            seen.add(n)
            unique.append(n)
    return unique


@retry(wait=_retry_wait, stop=_retry_stop)
def rewrite_query(question, history=[]):
    """Rewrite the user's question to be a more specific question that is more likely to surface relevant content in the Knowledge Base."""
    message = f"""
You are in a conversation with a user, answering questions about the company Insurellm.
You are about to look up information in a Knowledge Base to answer the user's question.
This is the history of your conversation so far with the user:
{history}
And this is the user's current question:
{question}
Respond only with a short, refined question that you will use to search the Knowledge Base.
It should be a VERY short specific question most likely to surface content. Focus on the question details.
IMPORTANT: Respond ONLY with the precise knowledgebase query, nothing else.
"""
    message = [SystemMessage(content=message)]
    resp = llm.invoke(message)
    return resp.content


def fetch_context_unranked(query) -> list[Document]:
    return retriever.invoke(query, k=RETRIEVAL_K)


def merge_chunks(chunks, reranked):
    merged = chunks[:]
    existing = [chunk.page_content for chunk in chunks]
    for chunk in reranked:
        if chunk.page_content not in existing:
            merged.append(chunk)
    return merged


@retry(wait=_retry_wait, stop=_retry_stop)
def rerank(question, chunks):
    system_prompt = """
You are a document re-ranker.
You are provided with a question and a list of relevant chunks of text from a query of a knowledge base.
The chunks are provided in the order they were retrieved; this should be approximately ordered by relevance, but you may be able to improve on that.
You must rank order the provided chunks by relevance to the question, with the most relevant chunk first.
Reply only with the list of ranked chunk ids, nothing else. Include all the chunk ids you are provided with, reranked.
"""
    user_prompt = f"The user has asked the following question:\n\n{question}\n\nOrder all the chunks of text by relevance to the question, from most relevant to least relevant. Include all the chunk ids you are provided with, reranked.\n\n"
    user_prompt += "Here are the chunks:\n\n"
    for index, chunk in enumerate(chunks):
        user_prompt += f"# CHUNK ID: {index + 1}:\n\n{chunk.page_content}\n\n"
    user_prompt += "Reply only with the list of ranked chunk ids, nothing else."
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    resp = llm_rerank.invoke(messages)
    order = _parse_rank_order(resp.content, len(chunks))
    # Fall back to original order if parsing yielded nothing
    if not order:
        return chunks
    return [chunks[i - 1] for i in order]


def fetch_context(original_ques: str):
    """
    Retrieve relevant context documents for a question.
    """
    rewritten_question = rewrite_query(original_ques)
    chunk1 = fetch_context_unranked(original_ques)
    chunk2 = fetch_context_unranked(rewritten_question)
    chunks = merge_chunks(chunk1, chunk2)
    reranked = rerank(original_ques, chunks)
    return reranked[:TOP_K]


def make_rag_messages(question, history, chunks):
    context = "\n\n".join(
        f"Extract from {chunk.metadata['source']}:\n{chunk.page_content}"
        for chunk in chunks
    )
    system_prompt = SYSTEM_PROMPT.format(context=context)
    return (
        [{"role": "system", "content": system_prompt}]
        + history
        + [{"role": "user", "content": question}]
    )


def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question


@retry(wait=_retry_wait, stop=_retry_stop)
def answer_question(question: str, history: list[dict] = []):
    """
    Answer the given question with RAG;
    """
    chunks = fetch_context(question)
    messages = make_rag_messages(question, history, chunks)
    response = llm.invoke(messages)
    return response.content, chunks
