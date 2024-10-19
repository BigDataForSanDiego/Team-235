
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

huggingface_api_key = os.getenv("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")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.set_page_config(page_title="Blood Report Analysis Chatbot", layout="centered")
## set up Streamlit 
st.title("AI Blood Report Analysis Assistant")
st.write("Upload your blood reports and receive insights based on the data. This tool is designed to help analyze blood test results and provide suggestions. If you notice any abnormal values, we recommend consulting a healthcare professional.")

## Input the Groq API Key
api_key=st.text_input("Enter your Groq API key:",type="password")

## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="Gemma2-9b-It")

    ## chat interface

    session_id=st.text_input("Session ID",value="default_session")
    ## statefully manage chat history

    if 'store' not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Upload your blood report",type="pdf",accept_multiple_files=True)
    ## Process uploaded  PDF's
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Split and create embeddings for the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma")
        retriever = vectorstore.as_retriever()    

        contextualize_q_system_prompt = (
    "You are an AI assistant tasked with analyzing blood reports and providing concise, non-diagnostic insights based on the data. "
    "Explain any abnormal values, suggest possible causes, and offer recommendations for lifestyle changes to improve the user's health. "
    "If the information is unclear or beyond your scope, remind the user to consult a healthcare professional. "
    "You are only allowed to analyze the provided blood reports and cannot answer questions that do not pertain to these reports."
)

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

        ## Answer question

        # Answer question
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
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            #st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")











