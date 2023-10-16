# Import streamlit for app dev
import streamlit as st

# Import transformer classes for generaiton
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
# Import torch for datatype attributes 
import torch
# Import the prompt wrapper...but for llama index
from llama_index.prompts.prompts import SimpleInputPrompt
# Import the llama index HF Wrapper
from llama_index.llms import HuggingFaceLLM
# Bring in embeddings wrapper
from llama_index.embeddings import LangchainEmbedding
# Bring in HF embeddings - need these to represent document chunks
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# Bring in stuff to change service context
from llama_index import set_global_service_context
from llama_index import ServiceContext
# Import deps to load documents 
from llama_index import VectorStoreIndex, download_loader
from pathlib import Path

# Define variable to hold llama2 weights naming 
name = "meta-llama/Llama-2-70b-chat-hf"
# Set auth token variable from hugging face 

auth_token = st.sidebar.text_input('Ingresa tu Hugging Face Auth Token aquí')

if auth_token:

    #auth_token = "YOUR HUGGING FACE AUTH TOKEN HERE"

    @st.cache_resource
    def get_tokenizer_model():
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)

        # Create model
        model = AutoModelForCausalLM.from_pretrained(name, cache_dir='./model/'
                                , use_auth_token=auth_token, torch_dtype=torch.float16, 
                                rope_scaling={"type": "dynamic", "factor": 2}, load_in_8bit=True) 

        return tokenizer, model
    tokenizer, model = get_tokenizer_model()

    # Create a system prompt 
    system_prompt = """<s>[INST] <<SYS>>
     Eres un asistente útil, respetuoso y honesto. Responde siempre de la manera 
     más útil posible. Tus respuestas no deben incluir contenido dañino, 
     poco ético, racista, sexista, tóxico, peligroso o ilegal.
     
     Si una pregunta no tiene ningún sentido o no es objetivamente coherente, solicita aclaración
     en lugar de responder algo no es correcto. Si no sabes la respuesta
     a una pregunta, por favor no compartas información falsa. Solo comenta: " No se la respuesta ".

     Tu objetivo es proporcionar respuestas relacionadas con el contenido legal del texto proporcionado.<</SYS>>
    """
    # Throw together the query wrapper
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    # Create a HF LLM using the llama index wrapper 
    llm = HuggingFaceLLM(context_window=4096,
                        max_new_tokens=256,
                        system_prompt=system_prompt,
                        query_wrapper_prompt=query_wrapper_prompt,
                        model=model,
                        tokenizer=tokenizer)

    # Create and dl embeddings instance  
    embeddings=LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Create new service context instance
    service_context = ServiceContext.from_defaults(
        chunk_size=1024,
        llm=llm,
        embed_model=embeddings
    )
    # And set the service context
    set_global_service_context(service_context)

    # Download PDF Loader 
    PyMuPDFReader = download_loader("PyMuPDFReader")
    # Create PDF Loader
    loader = PyMuPDFReader()
    # Load documents 
    documents = loader.load(file_path=Path('./data/annualreport.pdf'), metadata=True)

    # Create an index - we'll be able to query this in a sec
    index = VectorStoreIndex.from_documents(documents)
    # Setup index query engine using LLM 
    query_engine = index.as_query_engine()

    # Create centered main title 
    st.title('Asistente legal Llama2')
    # Create a text input box for the user
    prompt = st.text_input('Ingresa tu pregunta aquí')

    # If the user hits enter
    if prompt:
        response = query_engine.query(prompt)
        # ...and write it out to the screen
        st.write(response)

        # Display raw response object
        with st.expander('Respuesta'):
            st.write(response)
        # Display source text
        with st.expander('Texto fuente'):
            st.write(response.get_formatted_sources())

