import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-4.1-nano-2025-04-14")

prompt_template = PromptTemplate(
input_variables=["text"],
template="You are an Email Writing Assistant. The User will provide you with a text. You have to generate a professional email based on the text provided. The text is as follows: {text}"
)

chain = LLMChain(llm=llm, prompt=prompt_template)

st.title("Smart Email Writer")

text = st.text_area("Enter your text here: ")

if st.button("Generate Email"):
    if(text.strip() == ""):
        st.error("Text field is empty")
    else:
        response = chain.run(text)
        st.subheader("Generated Email:")
        st.write(response)
