#dont use stream while giving the response back to llm (json format)
# reduces the waiting time

import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import time

load_dotenv()

#ROLE:
#TASK
#CONSTRAINT
#OUTPUT FORMAT
#Example
#FALLBACK

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API KEY NOT FOUND!")

client = Groq(api_key=API_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role" : "system",
            "content":"""
            You are Engineering Professor assistant. 
            Give answer to the specific question asked by the client.
            Do not invent answer or give vague response
            make the answer more readable 
            If you dont know the answer say "Ask different question"
            """
        },
        {
            "role": "user",
            "content": "How does internet work?",
        }
    ],
    stream=True
)


for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
        time.sleep(0.02)