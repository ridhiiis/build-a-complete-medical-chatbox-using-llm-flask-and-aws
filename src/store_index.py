from encodings.idna import dots

from dotenv import load_dotenv
import os
from src.helper import load_pdf_files, filter_to_minimal_docs,text_split,download_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

import os

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
OPEN_ROUTER_API_KEY= os.getenv("OPEN_ROUTER_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPEN_ROUTER_API_KEY
os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"


extracted_data = load_pdf_files("C:\\Users\\ridhi\\medicalchatbox\\build-a-complete-medical-chatbox-using-llm-flask-and-aws\\data")
docs = load_pdf_files("C:\\Users\\ridhi\\medicalchatbox\\build-a-complete-medical-chatbox-using-llm-flask-and-aws\\data")
minimal_docs = filter_to_minimal_docs(docs)
text_chunks = text_split(minimal_docs)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)


index_name = "medical-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name = index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws" , region="us-east-1")
    )

    index = pc.Index(index_name)

# load Existing index
from langchain_pinecone import PineconeVectorStore

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)