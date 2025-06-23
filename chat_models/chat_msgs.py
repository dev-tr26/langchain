from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id = "microsoft/Phi-3-mini-4k-instruct",
    task ="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

llm.invoke()

messages = [
    SystemMessage("You are an expert in social media marketing content strategy", "you provide creative marketing/ad ideas"),
    HumanMessage("give short tip on engaing posts of instagram", "give ad/reel ideas to promot small buisness/shops"), 
]