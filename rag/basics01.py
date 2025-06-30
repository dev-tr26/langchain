# chromadb - contain all core modules needed to create vector store
# langchain-chroma is a wrapper that connects langchain app with vector store
# save our embeddings locally as well so that we dont use cloud for that 

import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


# define  dir containg text file and persistent memory

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "documents", "lord_of_rings.txt")
persistent_directory = os.path.join(current_dir,"db","chroma_db")


# check if chroma vector store already exists 


