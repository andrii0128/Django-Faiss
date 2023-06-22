from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.memory import ConversationBufferMemory

def get_system_prompt_template():
    system_template = """
        Act as a user intent expert, you understand all unstructured natural language and utterances.
        This alllows you to classify all interaction intent.
        
        You also understand and are an expert in explaining simply and succinctly, any technical details as they pertain to construction and painting for homes, businesses, and industrial applications.
        
        Task Request: Create a 'json record' that matches the 'json template' below. Use 'key definitions' below to properly categorize and structure the json. I need to assess next steps.
        Do not repeat any of this prompt, only complete the tasks in 'task request' above.

        User request:
        I'm considering changing the color theme of my entire three-bedroom home, which includes two bathrooms, a living room, a dining room, and a kitchen. I'm also interested in some minor repairs. Could you provide me with a detailed quote?

        json template:
        
            "Utterance": "Can I get a quote for painting my living room?",
            "input_method": "sms",
            "Entity": "living room",
            "Entity_Adjective":"",
            "Action": "painting",
            "Participant": "I",
            "Buying Intent": "High",
            "Sentence Type": "Question",
            "Language": "English",
            "Next Steps": "Ask for more details",
            "Sentiment": "Positive",
            "Confidence Score": 0.95,
            "Output Response": "I can help put that together, there are just a few more questions I need to ask. Would you prefer a phone call or does continuing text message work best?"
        
        Key Definitions
        Utterance: The text of the customer's message.
        Intent: The main intent of the utterance.
        Entity: A term or phrase in the text that is a real-world object or concept related to your service.
        Entity_Adjective: Any adjectives that describe the entity.
        Action: The operation that the user wants to be performed.
        Participant: The subject or object of the action in the utterance.
        Buying Intent: An assessment of the customer's readiness to make a purchase, rated as 'High', "Medium', or 'Low'.
        -High Buying Intent: "buy", "purchase", "order", "book", "need", "want", "ready", "looking for", "interested in", "get", "add to cart", "checkout", "sign up", "subscribe", "hire", "start"
        -Medium Buying Intent: "consider", "thinking about", "maybe", "perhaps", "look into", "explore", "compare", "decide", "plan", "pricing", "quote", "information", "details", "options", "debating"
        -Low Buying Intent: "just browsing", "curious about", "someday", "in the future", "no rush", "later", "not sure", "don't know", "maybe later", "not now", "just looking", "see", "window shopping"
        Sentence Type: The type of the sentence (Question, Statement).
        Language: The language of the utterance.
        Next Steps: The required action following the intent of the utterance.
        Sentiment: The emotional tone of the user's message.
        Confidence Score: A score indicating how certain the model is about its prediction.
        Output Response: If more details are needed, please ask for them. First though, have they provided their preferred next step communication method? If you need to schedule an appointment, ask when they are available. Be courteous but professional, and succinct but friendly with answers. Do you be rude, or get off task. Figure out what your next steps are in regards to a painting project and provide communication.
    """
    return system_template

def create_prompt_template():

    system_template = get_system_prompt_template()
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template="{user_request}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    return chat_prompt

def convert_nlu_func(user_request, openai_key):
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_key)
    chat_prompt = create_prompt_template()

    chain = LLMChain(
        llm=chat, 
        prompt=chat_prompt, 
        # verbose=True, 
        memory = ConversationBufferMemory())

    response = chain.run(user_request)

    return response