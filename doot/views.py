from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .utils.convert_nlu_into_json import convert_nlu_func
from .utils.get_embedding import get_embedding_func
from .utils.similar_search import similar_search_func

from .models import Utterance
from langchain.schema import Document

import json
import environ

env = environ.Env()
environ.Env.read_env()

openai_api_key = env('OPENAI_API_KEY')

@api_view(['GET', 'POST'])
def convert_nlu_into_json(request):
    if request.method == 'GET':
        user_request = request.query_params.get("user_request")
        try:
            query_result = convert_nlu_func(user_request, openai_api_key)
        except:
            query_result = "Sorry! conversion failed!"
        
        json_result = json.loads(query_result)
        return Response(json_result)

@api_view(['GET'])
def get_embedding(request):
    if request.method == 'GET':
        user_query = request.query_params.get("user_query")
        embedding = get_embedding_func(user_query, openai_api_key)
        return Response(embedding)
    
@api_view(['GET'])
def similar_search(request):
    if request.method == 'GET':
        utterance = request.query_params.get("utterance")
        docs = []
        dataset = Utterance.objects.all()
        for data in dataset:
            item = Document(page_content=data.utterance, metadata=dict(intent=data.intent_category))
            docs.append(item)

        result = similar_search_func(docs, utterance, openai_api_key)
        response = f"Utterance:{result.page_content}, Metadata:{result.metadata}"
        return Response(response)
