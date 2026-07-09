# the system prompt is the relation of the user to the llm 
# temperature is the lvl of randomness (creativity) of llm, 0 = safe answer,  1 = creative, 2= very creative 
# llm hallucinates when it does not know the answer, tries to guess it

import os
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key=API_KEY)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role" : "system",
            "content": "You are a brand manager who suggests name for my company. name should be in one word. suggest only one name"
        },
        {
            "role" : "user",
            "content": "Suggest a name for my food delivery company"
        }
    ],
    temperature=2
)

print("#"*100)

print(response.choices[0].message.content)

# for temperature = 0, Deliva
# for temperature = 1, Flavio
# for temperature = 2, Biteo