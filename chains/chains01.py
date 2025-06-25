from dotenv import load_dotenv 
import os
from langchain.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "MiniMaxAI/MiniMax-M1-80k",
    max_new_tokens= 512,
    temperature =0.7,
)

model = ChatHuggingFace(llm=llm)

#cdefine prompt templates (no need for seperate runnable chains)
# we are doing tuples way coz we want sys,human,ai msg

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
        ("human","Tell me {fact_count} facts."),
    ]
)

# connect different tasks together so that we dont need to call llm.invoke again
# create combined chain using langchain expression language (LCEL)

chain = prompt_template | model | StrOutputParser()
# chain = prompt_template | model 

response = chain.invoke({"animal":"cat", "fact_count":2})

print(response)