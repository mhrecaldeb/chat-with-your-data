import sys
import os
import datetime

import openai

import streamlit as st
from io import StringIO
from PyPDF2 import PdfReader

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.document_loaders import TextLoader

from langchain.document_loaders import PyPDFLoader

from langchain.chat_models import ChatOpenAI

from langchain.chains import RetrievalQA

import datetime

texto_extraido = ""

def main():

    current_date = datetime.datetime.now().date()
    if current_date < datetime.date(2023, 9, 2):
        llm_name = "gpt-3.5-turbo-0301"
    else:
        llm_name = "gpt-3.5-turbo"
    print(llm_name)


    # encabezado de streamlite
    st.title("Aplicación inteligente para profesionales del Derecho: ¡consulta tus documentos no los leas!")

    st.sidebar.title('Ingresa tu OpenAI API KEY')

    st.write("""
    ### Consulta a los documentos PDF

    Esta aplicación te permite preguntar a tus documentos!

    Basado en el curso corto de Deeplearning.ai "Langchain chat with your data" y el "Curso NLP" de Hugging Face
    """)

    key_entered = st.sidebar.text_input('Ingresa tu api key aquí')

    if key_entered:
        os.environ['OPENAI_API_KEY'] = key_entered
        openai.api_key  = os.environ['OPENAI_API_KEY']

        st.info("""
                [Langchain Short Courses](https://learn.deeplearning.ai/langchain-chat-with-your-data)
                
                [NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt)
                        
                """)

        #  Load PDF
        text = False
        text = upload_information()

        # Split text
        if text:
            qa_chain_builded = create_qa_chain(text, llm_name)

            # Crear un título central 
            st.title('Conversa con tus documentos')

            # Crear una caja de entrada de texto
            question = st.text_input('¿Qué quieres preguntar sobre el / los documentos?:')

            #question = input("¿Qué quieres preguntar sobre el / los documentos?: \n")


            # If the user hits enter
            if question:
                result = qa_chain_builded({"query": question})
                # ...and write it out to the screen
                st.write("""
                    ### Esta es la respuesta
                        """)
                st.write(result["result"])

                # Display raw response object
                with st.expander('Response Object'):
                    st.write(result)

# Función para extraer la información
def upload_information():
    """ This function run a code to upload a group of PDF files, extract the text from them
        and return a text object to be used in the following parts of the program"""
    global texto_extraido
    # Load PDF

    st.write("""
                ### Selecciona los PDF que quieres usar
    
                            """)
    
    uploaded_files = st.file_uploader('Choose your .pdf file', type="pdf", accept_multiple_files=True)
        
    # extract text from files

    if uploaded_files:
        
        texto_extraido = ""
        for uploaded_file in uploaded_files:
            st.write("filename:", uploaded_file.name)
            pdf_reader = PdfReader(uploaded_file)
            for page in pdf_reader.pages:
                texto_extraido += page.extract_text()

    return(texto_extraido)

def create_qa_chain(text_recibido, llm_nombrado = "gpt-3.5-turbo"):
    """ This function receives the text that was generated from PDF files, 
     split it, create the embeddings, and vectordb in memory, builds the prompt
      amd return a RetrivalQA object using an llm model """
    
    # Split text

    if text_recibido is not None:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        splits = text_splitter.split_text(text_recibido)
        embedding = OpenAIEmbeddings()


        vectordb = FAISS.from_texts(
            splits,
            embedding
        )

        ### Retrieval QA
        llm = ChatOpenAI(model_name=llm_nombrado, temperature=0)

        # Build prompt
        from langchain.prompts import PromptTemplate
        template = """Usa la siguiente pieza de contexto para responder la pregunta al final. Si no sabes la respuesta solo di, 
                    "Yo no se la respuesta a esa pregunta", no trates de inventar una respuesta. Usa tres frases máximo. Mantén la respuesta concisa. Siempre, después de un salto de línea, di: 
                    " Gracias por preguntar", al final de la respuesta. 
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
    return(qa_chain)


if __name__ == '__main__':
    main()