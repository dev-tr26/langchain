# parallel chains running and want output from any particular chain only based on certain conditions 

# user feedback -> +ve ->result
#               -> -VE 
#               -> neutral 


from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id ="MiniMaxAI/MiniMax-M1-80k", 
    temperature = 0.7,
    max_new_tokens=512,
)


model = ChatHuggingFace(llm =llm)

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant"),
        ("human","Generate a thank you note for this positive feedback: {feedback}."),
    ]
) 


negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant"),
        ("human","Generate a response for this negative feedback: {feedback}."),
    ]
) 

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant"),
        ("human","Generate a request for more details for this neutral feedback: {feedback}."),
    ]
) 

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assitant"),
        ("human"," generate a message to escalate this feedback to a human agent: {feedback}."),
    ]
) 


# feedback classification template

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant"),
        ("human", "Classify the sentiment of this feedback as positive, negative ,neutral, or escalate : {feedback}.")
    ]
)


# define runnable branches for handling feedback

branches = RunnableBranch (
    # +ve feedback chain
    (
        lambda x: "positive" in x,
        positive_feedback_template | model | StrOutputParser() 
    ),
    
    # negative feedback chain
    (
        lambda x: "negative" in x,
        negative_feedback_template | model | StrOutputParser() 
    ),
    
    # neutral feedback chain
    (
        lambda x: "neutral" in x,
        neutral_feedback_template | model | StrOutputParser() 
    ),
    
    # escalate feedback chain
     escalate_feedback_template | model | StrOutputParser()
    
)


# create classification chain
classification_chain = classification_template | model | StrOutputParser()

# combine classification and response generation in chain 
chain = classification_chain | branches

# run chain with example review 

# Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
# Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
# Neutral review - "The product is okay. It works as expected but nothing exceptional."
# Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"


review = "The product is terrible. It broke after just one use and the quality is very poor."

result = chain.invoke({"feedback": review})

print(result)

