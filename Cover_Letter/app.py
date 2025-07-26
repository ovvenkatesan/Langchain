import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-4.1-nano-2025-04-14")

prompt_template = PromptTemplate(
    input_variables=["resume_text", "job_title", "company_name"],
    template="""
        You are a expert career coach and writer. 
        Write a professional, personalized cover letter for this role  
        Job Title : {job_title} 
        Company Name: {company_name}
        Resume:{resume_text}
        
        Keep the tone farmal and professional, and avoid any unnecessary information.
        Highlight relevant skills, experience, and accomplishments and enthusisam for the role.

        """
)

coverLetterChain = LLMChain(llm=llm, prompt=prompt_template)

# Steamlit UI
st.title("Cover Letter Generator")

uploaded_file = st.file_uploader("Upload your resume", type=["pdf", "docx", "txt"], accept_multiple_files=False)
job_title = st.text_input("Enter your job title")
company_name = st.text_input("Enter your company name (Optional)")

resume_text = ""

if(st.button("Generate Cover Letter")):
    if not uploaded_file or not job_title :
        st.error("Please upload your resume and enter your job title")
    else:
        if(uploaded_file.name.endswith(".pdf")):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            resume_text = " ".join([page.extract_text() for page in pdf_reader.pages])
        elif(uploaded_file.name.endswith(".txt")):
            resume_text = uploaded_file.getvalue().decode('utf-8')
            resume_text = resume_text.replace("\n", " ")
            resume_text = resume_text.replace("\r", " ")
        elif(uploaded_file.name.endswith(".docx")):
            resume_text = uploaded_file.getvalue().decode('utf-8')
            resume_text = resume_text.replace("\n", " ")
            resume_text = resume_text.replace("\r", " ")
        else:
            st.error("Invalid file type")
            
    if(resume_text not in ["", None]):
        response = coverLetterChain.run(resume_text=resume_text, job_title=job_title, company_name=company_name)
        st.subheader("Generated Cover Letter:")
        st.write(response)

        
  
