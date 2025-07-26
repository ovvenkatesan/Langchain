import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.2, model_name="gpt-4.1-nano-2025-04-14")

embeddings = OpenAIEmbeddings()

st.title("Ask anything from PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    raw_text = ""
    try:
        document_loader = PdfReader(uploaded_file)
        raw_text = "\n".join(page.extract_text() for page in document_loader.pages)
        st.write("Text extracted successfully!")
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
    
    if(not raw_text.strip()):
        st.error("No text extracted from PDF")
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(raw_text)

        st.write("Chunks created successfully!")

        if(len(chunks) > 0):
            st.write("Number of chunks: ", len(chunks))
            vectorstore = FAISS.from_texts(chunks, embeddings)

            qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

            question = st.text_input("Ask a question:")

            if question:
                with st.spinner("Processing question..."):
                    answer = qa.run(question)
                st.subheader("Answer:")
                st.write(answer)
            else:
                st.error("No question provided")

        else:
            st.error("No chunks created")




