import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(api_key=os.getenv('OPENAI_API_KEY'), temperature=0.5, model_name="gpt-4.1-nano-2025-04-14")

summary_prompt = PromptTemplate(
    input_variables=["transcript"],
    template = """

You are an expert summarizer. Given a long text transcript, summarize it into a concise summary.    

Here is the transcript: {transcript}

Please generate a clean, concise summary of the main topics and points discussed in the transcript.

"""
)


summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

st.title("Youtube Video Summarizer")

video_url = st.text_input("Enter the Youtube Video URL: ")

def get_video_id(url):
    """Extract video ID from the URL"""
    parsed_url = urlparse(url)
    if parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]
    elif parsed_url.hostname in ("www.youtube.com", "youtube.com"):
        query = parse_qs(parsed_url.query)
        return query.get("v", [None])[0]
    return None

if st.button("Summarize"):
    if not video_url:
        st.error("Please enter a URL")
    else:
        video_id = get_video_id(video_url)
        if not video_id:
            st.error("Invalid URL")
        else:
            try:
                ytt_api = YouTubeTranscriptApi()
                transcript = ytt_api.fetch(video_id)
                transcript_text = " ".join([snippet.text for snippet in transcript.snippets])
                summary = summary_chain.run(transcript=transcript_text)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error fetching transcript: {e}")
            
            

