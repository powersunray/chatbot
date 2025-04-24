import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_together import TogetherEmbeddings

from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain


from langchain_together import Together

import os
from dotenv import load_dotenv

load_dotenv()


#Upload PDF files
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")
    
#Extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)
        
    #Break it into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)
    
    #generating embeddings
    embeddings = TogetherEmbeddings(
        api_key=os.getenv("TOGETHER_AI_API_KEY"),
        model="togethercomputer/m2-bert-80M-32k-retrieval"
    )

    #creating vector store - FAISS: Facebook AI Similarity Search
    vector_store = FAISS.from_texts(chunks, embeddings)

    #get user question
    user_question = st.text_input("Type your question here")
    
    #do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        #st.write(match)
    
        llm = Together(
            api_key=os.getenv("TOGETHER_AI_API_KEY"),
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            temperature=0,
            max_tokens=1000
        )
    
        #output results
        #chain -> take the question, get relevant document, pass it to the LLM, generate the output
        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents = match, question = user_question)
        st.write(response)
