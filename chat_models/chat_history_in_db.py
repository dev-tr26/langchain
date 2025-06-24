from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from google.cloud import firestore
from langchain_google_firestore import FirestoreChatMessageHistory
import os 
'''

# example store source : https//python.langchain.com/v0.2/docs/integration/memory/google_firestore/

# in prod grade app it is stored in cloud (chathistory )

# use firebase db firestore is doc based nosql db

steps to replace :
    1.0 create firebase acc , new project, firebase db
    1.1 pip install langchain_google_firestore 
    2.0 retrieve project id  (can ve muiltiple firebase projects )
    2.1 store project session id,collection name in .env
    2.3. install google cloud cli with your google on pc 
        - https://cloud.google.com/sdk/docs/install
        - authenticate google cloud cli with your google acc
        - https://console.cloud.google.com/docs/authentication/provide-credentials-adc#local-dev
        - set your default project to new firebase project u created 

    2.4. enable firestore api in googlee cloud console  
        - https://console.cloud.google.com/apis/enablelow?apiid=firestore.googleapis.comproject=crewai-automation

'''

load_dotenv()

PROJECT_ID = os.getenv("PROJECT_ID")
SESSION_ID = os.getenv("SESSION_ID")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")




print("initialising firestore client....")
client = firestore.Client(project=PROJECT_ID)

print("initialising firestore Chat message History")
chat_history = FirestoreChatMessageHistory(
    session_id=SESSION_ID,
    collection=COLLECTION_NAME,
    client=client,
) 

print("Chat history initialised.")
print(f'current Chat History : {chat_history.messages}')


# initialises chat model 

llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    temperature=0.6,
    task='text-generation',
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat = ChatHuggingFace(llm=llm, verbose=True) 

print("Start chatting with AI. Type 'exit' to quit.")

while True:
    human_input = input("User: ")
    if human_input.lower() == "exit":
        break
    
    chat_history.add_user_message(human_input)
    
    ai_response = chat.invoke(chat_history.messages)
    chat_history.add_ai_message(ai_response.content)
    
    print(f"AI: {ai_response}")