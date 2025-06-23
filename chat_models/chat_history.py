from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()



# run loacally 
# First, create a HuggingFaceHub instance
# how to communicate model from terminal
# store conv/chat history in memory in local var 


llm4 = HuggingFaceEndpoint(repo_id ="microsoft/Phi-3-mini-4k-instruct",
                           task ="text-generation",
                        #    model_kwargs={"temperature":0.6,
                        #                  "max_new_tokens":64}
                        )

chat_history =[]

system_message = SystemMessage(content = "you are a helpful ai assistant.")
chat_history.append(system_message)

while True:
    query =input("You: ")
    if query.lower() == "exit":
        break
    
    chat_history.append(HumanMessage(content=query))
    
    result = llm4.invoke(chat_history)
    response = result
    chat_history.append(AIMessage(content = response))
    
    print(f"AI: {response}")
    

print("-----Message History -----")
print("chat_history")


# -------------------------------------------

# in prod grade app it is stored in cloud (chathistory )

# use firebase db 
