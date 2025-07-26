from langchain import OpenAI, LLMChain, PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.7)


prompt_template = PromptTemplate(
    template="You are a helpful AI assistant. user Says: {user_input}. You have to generate a response : ",
    input_variables=["user_input"]
)

chain = LLMChain(llm=llm, prompt=prompt_template)

if __name__ == "__main__":
    user_input = input("User: ")
    response = chain.run(user_input=user_input)
    print("AI: " + response)
