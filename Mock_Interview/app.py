import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-4.1-nano-2025-04-14")

prompt_template = PromptTemplate(
    input_variables=["role", "job_description"],
    template="""
        You are an experienced interviewer conducting a mock interview for the {role} job description at {job_description}. 
        Ask 5 technical **Technical** mock interview questions for this role.  
        Only Include the questions that test technical skills, knowledge or problem solving.
        Do not include situational or behavioural questions.

        For each question, also provide clear, strong sample answers.

        Number them 1 to 5, and format like this:

        1. Question: [Your question here]
        Answer: [Sample answer here]
        
        2. Question: [Your question here]
        Answer: [Sample answer here]
        
        Continue this format for all 5 questions.
        """
)

chain = LLMChain(llm=llm, prompt=prompt_template)

st.title("Mock Interview")

role = st.text_input("Job Role : example: Software Engineer, Data Scientist, etc.  :  ")
job_description = st.text_input("Job Description ( past here ) ")

if st.button("Start Interview"):
    if(role.strip() == "" or job_description.strip() == ""):
        st.error("Please enter a role and job_description name")
    else:
        response = chain.run(role=role, job_description=job_description)
        st.subheader("Interviewer Response:")
        st.write(response)
