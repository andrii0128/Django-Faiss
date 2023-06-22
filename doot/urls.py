from django.urls import path
from .views import convert_nlu_into_json, get_embedding, similar_search

urlpatterns = [
    path('get_json', convert_nlu_into_json, name="convert_nlu_into_json"),
    path('get_embedding', get_embedding, name="get_embedding"),
    path('similar_search', similar_search, name="similar_search"),
]