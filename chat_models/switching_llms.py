from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv


load_dotenv()

# messages = [
#     SystemMessage(conyent = "solve the following math problems"),
#     HumanMessage("what is the square of 55"),
    
# ]

# model = ChatAnthropic(model ="claude=3-opus-20240229")

# result = model.invoke(messages)
# print(f'ans from anthropic: {result.content}')
# -----------------------------------------------


# model = ChatGoogleGenerativeAI(model ="gemini-1.5-flash")
# result2 = model.invoke(messages)
# print(f'result from gemini: {result2.content}')

# -------------------------------------------------------

# run loacally 
# First, create a HuggingFaceHub instance
# how to communicate model from terminal
# store conv/chat history in memory in local var 


llm4 = HuggingFaceEndpoint(repo_id ="microsoft/Phi-3-mini-4k-instruct",
                           task ="text-generation",)

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