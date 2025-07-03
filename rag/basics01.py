# chromadb - contain all core modules needed to create vector store
# langchain-chroma is a wrapper that connects langchain app with vector store
# save our embeddings locally as well so that we dont use cloud for that 

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


# define  dir containg text file and persistent memory

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_rings.txt")
persistent_directory = os.path.join(current_dir,"db","chroma_db")


# check if chroma vector store already exists 
# if db is exists then only coz text to embedding is costly 

if not os.path.exists(persistent_directory):
    print("Persistent directory (db) does not exist. Initializing vector store....")

    # ensure text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")
    
    
    # read text content from file
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # split docs into chunks
    # chunk overlap = how many chars /tokens depending on splitter should be included in both end of one chain and beginning of next 
    # if chunk_overlap =0 then no overlap between chunks 
    # if its 100 then 100 tokens of one chunk will appear at beginning of next chunk . this helps preserve context across chunks
    # also helps in semantic understanding especially when working with large texts (books) chunking with overlap helps better understanding across chunks when querying or processing 
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50) 
    docs = text_splitter.split_documents(documents)
    
    # display info about split documents 
    print("\n --- Document Chunks Info ---")
    print(f'Number of document chunks: {len(docs)}')
    print(f"Simple chunk: \n{docs[0].page_content}\n")
    
    # create embeddings 
    print("\n ---- creating embeddings ----")
    embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
    )
    
    
    # create vector store and persist it autometically 
    print("\n --- creating vector store ---")
    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persistent_directory
    )
    
    print("\n--- Vector store created and saved locally. ---")

else:
    print("vector store already exists no need to initialize ")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=persistent_directory,
        embedding_function=embeddings
    )