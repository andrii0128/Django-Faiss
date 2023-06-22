from langchain.embeddings import OpenAIEmbeddings

def get_embedding_func(user_query, openai_key):
    
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    query_result = embeddings.embed_query(user_query)
    return query_result