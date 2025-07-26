import streamlit as st
from PyPDF2 import PdfReader
from openai import OpenAI
import requests
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')
serp_api_key = os.getenv('SERPAPI_API_KEY')

client = OpenAI()

@st.cache_resource
def load_pdf_create_index(pdf_path):
    docs = []
    for path in pdf_path:
        reader = PdfReader(path)
        text = " ".join([page.extract_text() for page in reader.pages])
        docs.append(text)
    chunks = []
    for doc in docs:
        for i in range(0, len(doc), 500):
            chunk = doc[i:i+500]
            if chunk.strip():
                chunks.append(chunk)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    vectors = model.encode(chunks)

    # Create text-embedding pairs for FAISS
    text_embeddings = list(zip(chunks, vectors))
    
    # Create a dummy embedding function (FAISS needs this but we already have embeddings)
    class DummyEmbeddings:
        def embed_documents(self, texts):
            return [model.encode(text) for text in texts]
        def embed_query(self, text):
            return model.encode(text)
    
    dummy_embeddings = DummyEmbeddings()
    index = FAISS.from_embeddings(text_embeddings, dummy_embeddings)

    return chunks, index, model

def retrieve(query, chunks, index, model, top_k=3):
    q_emp = model.encode([query])
    distances, indices = index.search(q_emp, top_k)
    return [chunks[i] for i in indices[0]]

def is_answer_sufficient(question,answer):
    prompt = f""" 
                query: {question}
                answer: {answer}
                Is the retrieved answer sufficient, accurate and complete? Reply with YES or NO and a short reson. """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def serp_search(query):
    url = f"https://serpapi.com/search?q={query}&api_key={serp_api_key}"
    response = requests.get(url)
    data = response.json()
    organic =  data["organic_results", []]

    snippets = []

    for result in organic:
        snippet = result.get("snippet") or result.get("title")
        if snippet:
            snippets.append(snippet)

    if snippets:
        return "\n".join(snippets)
    else:
        fallback_prompt = f""" the Web Search for the question : "{query}" returned no releant results 
         Please generate a Helpful answer to this question using your genral knowlege """
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": fallback_prompt}
            ]
        )
        return response.choices[0].message.content
    

def answer_query(query, chunks, index, model):
    retrieved = retrieve(query, chunks, index, model)
    combined = "\n\n".join(retrieved)

    verdict = is_answer_sufficient(query, combined)
    if verdict == "YES":
        return f""" ** From PDF: ""\n\n{combined}\n\n ""Vefified** {verdict} """
    else:
        web_search = serp_search(query)
        return f""" ** From Web: ""\n\n{web_search}\n\n ""Vefified** {verdict} """

st.title("Ask anything from PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], accept_multiple_files=True)

if uploaded_file is not None:
    chunks, index, model = load_pdf_create_index(uploaded_file)

    question = st.text_input("Ask a question:")

    if question:
        with st.spinner("Processing question..."):
            answer = answer_query(question, chunks, index, model)
        st.subheader("Answer:")
        st.write(answer)
