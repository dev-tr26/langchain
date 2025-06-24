from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


# ✅ Correct: define endpoint first
llm = HuggingFaceEndpoint(
    repo_id = "microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7,
)


# ✅ Correct: wrap in ChatHuggingFace
llm = ChatHuggingFace(llm=llm)

# this is just a plain string not a prompt template that we understand not langchain.
# template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skills} as a key strength. Keep it 4 lines max"

# result = llm.invoke(template)
# print(result)

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it 4 lines max"


prompt_template = ChatPromptTemplate.from_template(template)

# print(prompt_template)

# convert prompt to that langchain understands
# it creates list with just one human msg

'''

prompt = prompt_template.invoke({    
    "tone": "energetic",
    "company": "Samsung",
    "position" : "Ai engineer",
    "skill": "ai",
})


#to get just mail

result = llm.invoke(prompt)
print(result.content)     

print(result)

'''


# for passing conv as prompt 

messages = [
    ("system", "you are comedian who tells jokes about {topic}"),
    ("human", "Tell me a {joke_count} jokes."),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({"topic": "dark-comedy", "joke_count": "5"})


print("\n---- Prompt with System Human Messages (tuple)")
print(prompt)