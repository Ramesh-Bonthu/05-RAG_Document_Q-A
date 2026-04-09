## RAG Document Q&A with GROQ API and LLAMA3


#import libraries
import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

load_dotenv()

## load groq api
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
groq_api_key= os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

## prompt
prompt = ChatPromptTemplate.from_template(
    """
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the context
        <context>
            {context}
        </context>
        Question:{input}
    """
)

## method for vector embeddings
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.vectors = None
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name = "all-miniLM-L6-v2")
        st.session_state.docs = PyPDFDirectoryLoader("research_papers").load()
        st.session_state.text_spiltters = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=50)
        st.session_state.final_docs = st.session_state.text_spiltters.split_documents(st.session_state.docs[:50])
        st.session_state.vectordb = FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)

## Streamlit Interface
st.title("RAG Document Q&A with Groq and Hugging Face")

user_prompt = st.text_input("Enter the query from the research papers")

if st.button("Document Embeddings"):
    create_vector_embeddings()
    st.write("Vector store is ready.")

if user_prompt:
    doc_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vectordb.as_retriever()
    rag_chain = create_retrieval_chain(retriever,doc_chain)

    start = time.process_time()
    response = rag_chain.invoke({"input":user_prompt})
    print(f"Processing time : {time.process_time() - start}")

    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------------")


