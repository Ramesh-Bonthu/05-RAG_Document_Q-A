## Coversational RAG Q&A with PDF's uploads and chat history

# import libraries
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

load_dotenv()
# loading api keys
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
embedings = HuggingFaceEmbeddings(model_name="all-miniLM-L6-v2")

## Streamlit interface
st.title("Coversational RAG Q&A with PDF's uploads and chat history")
st.write("upload a pdfs and chat with their content")

api_key = st.text_input("Enter your Groq api key",type="password")

if api_key:
    llm = ChatGroq(groq_api_key = api_key,model="openai/gpt-oss-120b")

    ##session
    session_id = st.text_input("Session ID",value="default_session_id")

    if "store" not in st.session_state:
        st.session_state.store = {}
    
    uploaded_files = st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents = []

        for upload_file in uploaded_files:
            temppdf = f"./temp.pdf"

            with open(temppdf,"wb") as file:
                file.write(upload_file.getvalue())
                file_name = upload_file.name
            
            docs = PyPDFLoader(temppdf).load()
            documents.extend(docs)
        
        ## split and create documents for the embeddings
        text_splitters = RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
        final_docs = text_splitters.split_documents(documents)
        vectordb = FAISS.from_documents(final_docs,embedings)
        retriever = vectordb.as_retriever()

        contextalize_q_systemprompt = (
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contextalize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",contextalize_q_systemprompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm,retriever,contextalize_q_prompt)

        # question and answer prompt
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )

        document_chain = create_stuff_documents_chain(llm,qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,document_chain)

        ## chat history

        def get_session_history(session_id:str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            
            return st.session_state.store[session_id]

        conversation_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        ## user question
        user_input = st.text_input("Your Question :")

        if user_input:
            session_history = get_session_history(session_id)
            response = conversation_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable":{
                        "session_id":session_id
                    }
                }
            )

            st.write(st.session_state.store)
            st.write("Assistant : ",response['answer'])
            st.write("Chat History : ",session_history.messages)
else:
    st.warning("Please enter your Groq API key")




