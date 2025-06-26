from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda,RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain.schema.output_parser import StrOutputParser


load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M1-80k",
    temperature=0.5,
    max_new_tokens=213,
)

model = ChatHuggingFace(llm=llm)


# define prompt templates 

human_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system","You love psycology and neuroscience and tell facts about {humans}"),
        ("human","Tell me {count} facts."),
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ("system","You are a translator and convert the provided text to {language}."),
        ("human","Translate the following text to {language}: {text}"),
    ]
)


# define additional processing steps using runnable lambda

count_words = RunnableLambda(lambda x: f"word count: {len(x.split())}\n{x}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})

# create combined chain usong LCEL
# we can also add talk to twitter api and post it on x 

chain = human_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser()

# run the chain

result = chain.invoke({"humans": "girls", "count" : 3})

print(result)

