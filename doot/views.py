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
        # query_result = convert_nlu_func(user_request, openai_api_key)
        query_result = """{
            "Utterance": "My house has three rooms. Could you provide a good solution to make it more wonderful?",
            "Intent": "home improvement",
            "Entity": "house",
            "Entity_Adjective": "three rooms",
            "Action": "provide solution",
            "Participant": "you",
            "Buying Intent": "Medium",
            "Sentence Type": "Question",
            "Language": "English",
            "Next Steps": "Ask for more details about the rooms and what the customer is looking for in terms of improvement.",
            "Sentiment": "Positive",
            "Confidence Score": 0.92,
            "Output Response": "Of course, I'd be happy to help. Can you tell me a bit more about the rooms and what you're looking for in terms of improvement? Would you prefer to continue via text or would you like to schedule a call to discuss further?"
        }
        """
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
        # docs = []
        # dataset = Utterance.objects.all()
        # for data in dataset:
        #     item = Document(page_content=data.utterance, metadata=dict(intent=data.intent_category))
        #     docs.append(item)
        docs = [
            Document(page_content='Can you provide me an estimate for painting my living room?', metadata={'intent': 'Request_Quote'}),
            Document(page_content="I'm interested in a quote for repainting the exterior of my house.", metadata={'intent': 'Request_Quote'}),
            Document(page_content='How much would it cost to paint my kitchen cabinets?', metadata={'intent': 'Request_Quote'}),
            Document(page_content='I would like a proposal for a fresh coat of paint in my bedrooms.', metadata={'intent': 'Request_Quote'}),
            Document(page_content='Can you give me a price for painting my basement?', metadata={'intent': 'Request_Quote'}),
            Document(page_content="What's the cost to paint the ceiling in my hallway?", metadata={'intent': 'Request_Quote'}),
            Document(page_content='Can you provide a quote for painting my basement, including repairing some water damage?', metadata={'intent': 'Request_Quote'}),
            Document(page_content="I need an estimate for repainting my home's exterior and replacing a few damaged siding panels.", metadata={'intent': 'Request_Quote'}),
            Document(page_content="What's the cost for painting the interior of a 4-bedroom house?", metadata={'intent': 'Request_Quote'}),
            Document(page_content="Could you give me a quote to paint my home's exterior, including the garage?", metadata={'intent': 'Request_Quote'}),
            Document(page_content='I need a quote for painting my hallway and fixing some cracked plaster.', metadata={'intent': 'Request_Quote'}),
            Document(page_content='How much would you charge to paint my kitchen and repair some water stains on the ceiling?', metadata={'intent': 'Request_Quote'}),
            Document(page_content="What's the cost of painting the interior of a 2000 square foot house?", metadata={'intent': 'Request_Quote'}),
            Document(page_content='Can I get a quote for painting the exterior of my single-story house and repairing some rotten wood?', metadata={'intent': 'Request_Quote'}),
            Document(page_content="I'd like a quote for painting my master bedroom, including the closet.", metadata={'intent': 'Request_Quote'}),
            Document(page_content='Can you provide a cost estimate to paint my living room and dining room, along with some minor drywall repairs?', metadata={'intent': 'Request_Quote'}),
            Document(page_content='How much would it be to paint the exterior of my house and repair some chipped paint?', metadata={'intent': 'Request_Quote'}),
            Document(page_content='Could you estimate the cost to repaint all my exterior doors and window frames?', metadata={'intent': 'Request_Quote'}),
            Document(page_content='I need a quote for painting all the interior doors in my home.', metadata={'intent': 'Request_Quote'}),
            Document(page_content="What's the cost to paint my house's exterior, including repairing a few areas of damaged stucco?", metadata={'intent': 'Request_Quote'}),
            Document(page_content="Can you give me an estimate for painting my home's entire interior and repairing some wall cracks?", metadata={'intent': 'Request_Quote'}),
        ]

        result = similar_search_func(docs, utterance, openai_api_key)        
        # for item in docs:
        #     print(f"Document({item}),")
        # result = docs[0]
        response = f"Utterance:{result.page_content}, Metadata:{result.metadata}"
        return Response(response)
