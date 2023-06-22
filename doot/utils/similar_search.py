from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

def similar_search_func(docs, query, openai_key):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    db = FAISS.from_documents(docs, embeddings)
    docs = db.similarity_search(query)
    return docs[0]