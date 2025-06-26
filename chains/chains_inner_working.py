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

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","You love psycology and neuroscience and tell facts about {humans}"),
        ("human","Tell me {count} facts."),
    ]
)

# create individual runnables (steps in chain)

# invoke method --> replace placeholder val,convertss data to right format that can be send to llms
# that does format_prompt --> convert to llm understandable format

format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
parse_output = RunnableLambda(lambda x: x.content)

# create runnable sequence (equivalent to LECL chain)
# 1st and last runnable one , middle is list
chain = RunnableSequence(first=format_prompt, middle = [invoke_model], last = parse_output )

# run the chain

response = chain.invoke({"humans":"boys", "count":3})

# output 

print(response)