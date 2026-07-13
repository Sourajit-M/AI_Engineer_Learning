import os
from pathlib import Path
from dotenv import load_dotenv
import json
from groq import Groq
from pydantic import BaseModel

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("API KEY NOT FOUND!")

client = Groq(api_key=GROQ_API_KEY)

class Ticket(BaseModel):
    name: str
    email: str
    issue: str

schema = Ticket.model_json_schema()

system_prompt=f"""
Extract the personal information from the ticket strictly based on this schema and give a json output.
{schema}
"""

system_message = {
    "role" : "system",
    "content" : system_prompt
}

text="Hello My name is Pratyush. Yesterday I broke up with my girlfriend sheetal I have an iphone which is not working at all. My address is delhi. My email is abc@gmail.com. My contact number is 82134"
prompt=f"""
This is a customer ticket. Please extract the personal information from this.
{text}
"""

user_message = {
    "role" : "user",
    "content": prompt
}

messages = [user_message, system_message]

response = client.chat.completions.create(
    model= "llama-3.3-70b-versatile",
    messages= messages,
    response_format={
        "type": "json_object"
    }
)

answer = response.choices[0].message.content
print(answer)

df = json.loads(answer)
ticket = Ticket(**df)

print("#"*100)
print(ticket.name)
print(ticket.email)
print(ticket.issue)