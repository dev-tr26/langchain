import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# define persistent directory 

current_directory = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_directory, "db", "chroma_db")

# define the embedding model (should be same as model we used to embedd private data )
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# load the existing vectorstore with embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# define user query 
query = "where does Gandalf meet Frodo ?"

# retrieve relavent documents based on query
retriver = db.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs ={"k":3, "score_threshold":0.5},
)

relavent_docs = retriver.invoke(query)

# display relavent results with metadata
print("\n --- Relavent Docs ------")
for i , doc in enumerate(relavent_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source :{doc.metadata.get('source','Unkown')}\n")