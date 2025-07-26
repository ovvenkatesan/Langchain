import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-4.1-nano-2025-04-14")

prompt_template = PromptTemplate(
    template="""You are a professional coding assistant. Help the User with the following Task: {user_input}. Provide clean, well-commented code and explaination if needed: """,
    input_variables=["user_input"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

st.title("Code Assistant")

code_task = st.text_area("Describe your coding task here: ")

if st.button("Generate Code"):
    if(code_task.strip() == ""):
        st.error("Please provide a task description.")
    else:
        response = chain.run(user_input=code_task)
        st.subheader("Assistant Response:")
        st.code(response, language="python")

