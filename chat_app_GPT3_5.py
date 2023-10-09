# Import streamlit for app dev

import sys
import os
import datetime

import openai

import streamlit as st


#openai.api_key  = os.environ['OPENAI_API_KEY']
#openai_api_key = input("Ingresa tu API KEY de OpenAI: \n")
#openai_api_key = ""

os.environ['OPENAI_API_KEY'] = input("Ingresa tu API KEY de OpenAI: \n")
openai.api_key  = os.environ['OPENAI_API_KEY']


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import PyPDFLoader

from langchain.chat_models import ChatOpenAI

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)


# Load PDF
loaders = [
    # Duplicate documents on purpose - messy data
    PyPDFLoader("./Data/Resolucion_no_001_NG_Dinarp_2023.pdf"),
    PyPDFLoader("./Data/Resolucion_no_002_NG_Dinarp_2023.pdf"),
    PyPDFLoader("./Data/Resolucion_no_003_NG_Dinarp_2023.pdf"),
    PyPDFLoader("./Data/Resolucion_no_008_NG_Dinarp_2023.pdf")
]
docs = []
for loader in loaders:
    docs.extend(loader.load())


# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1500,
    chunk_overlap = 150
)


splits = text_splitter.split_documents(docs)

## persist_directory = './Data/chroma/'  # se podria descomentar para tratar chroma en streamlit

# a continuación se podría descomentar para usar chroma con el persist_directory
""""
for root, dirs, files in os.walk(persist_directory):
    for file in files:
        if file.endswith('.pdf'):
            os.remove(os.path.join(root, file))
"""

embedding = OpenAIEmbeddings()
## vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)  # se podría descomentar para usar chroma en streamlit
vectordb = FAISS.from_documents(
    splits,
    embedding
)

### Retrieval QA
llm = ChatOpenAI(model_name=llm_name, temperature=0)

# Build prompt
from langchain.prompts import PromptTemplate
template = """Usa la siguiente pieza de contexto para responder la pregunta al final. Si no sabes la respuesta solo di, "Yo no se", no trates de inventar una respuesta. Usa tres frases máximo. Mantén la respuesta concisa. Siempre di: "Gracias por preguntar", al final de la respuesta. 
{context}
Pregunta: {question}
Respuesta útil:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


# Run chain
from langchain.chains import RetrievalQA

question = input("¿Qué quieres preguntar sobre el / los documentos?: \n")

#"¿Qué señala el artículo 86 de la Ley de Transformación Digital y Audiovisual?"

qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=vectordb.as_retriever(),
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})


result = qa_chain({"query": question})


result["result"]
print(result["result"])
