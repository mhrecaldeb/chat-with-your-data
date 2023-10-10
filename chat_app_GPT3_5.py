# Import streamlit for app dev

import sys
import os
import datetime

import openai

import streamlit as st


#openai.api_key  = os.environ['OPENAI_API_KEY']
#openai_api_key = input("Ingresa tu API KEY de OpenAI: \n")
#openai_api_key = ""

#os.environ['OPENAI_API_KEY'] = input("Ingresa tu API KEY de OpenAI: \n")
#openai.api_key  = os.environ['OPENAI_API_KEY']


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import PyPDFLoader

from langchain.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)


# encabezado de streamlite
st.sidebar.header('Ingresa tu OpenAI API KEY')

st.sidebar.write("""
### Consulta a los documentos PDF

Esta aplicación te permite preguntar a tus documentos!

Basado en el curso corto de Deeplearning.ai "Langchain chat with your data" y el Github Llama2RAG de Nick Nochnack
""")

st.sidebar.markdown("""
[Langchain Short Courses](https://learn.deeplearning.ai/langchain-chat-with-your-data)
                    
[Llam2RAG](https://github.com/nicknochnack/Llama2RAG)
                    
""")


key_entered = st.sidebar.text_input('Ingresa tu api key aquí')

if key_entered:
    os.environ['OPENAI_API_KEY'] = key_entered
    openai.api_key  = os.environ['OPENAI_API_KEY']


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
    #""""
    #for root, dirs, files in os.walk(persist_directory):
    #    for file in files:
    #        if file.endswith('.pdf'):
    #            os.remove(os.path.join(root, file))
    #"""

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
    template = """Usa la siguiente pieza de contexto para responder la pregunta al final. Si no sabes la respuesta solo di, 
                "Yo no se", no trates de inventar una respuesta. Usa tres frases máximo. Mantén la respuesta concisa. Siempre di: 
                "Gracias por preguntar", al final de la respuesta. 
    {context}
    Pregunta: {question}
    Respuesta útil:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)


    # Run chain
    #from langchain.chains import RetrievalQA

    #"¿Qué señala el artículo 86 de la Ley de Transformación Digital y Audiovisual?"

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})



    # Crear un título central 
    st.title('Conversa con tus documentos')

    # Crear una caja de entrada de texto
    question = st.text_input('¿Qué quieres preguntar sobre el / los documentos?:')

    #question = input("¿Qué quieres preguntar sobre el / los documentos?: \n")


    # If the user hits enter
    if question:
        result = qa_chain({"query": question})
        # ...and write it out to the screen
        st.write("""
            ### Esta es la respuesta
                """)
        st.write(result["result"])

        # Display raw response object
        with st.expander('Response Object'):
            st.write(result)
        # Display source text
        #with st.expander('Source Text'):
        #    st.write(result.get_formatted_sources())



    #result = qa_chain({"query": question})
    #result["result"]
    #print(result["result"])
