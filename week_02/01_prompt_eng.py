import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key kaha hai bhai")

client=Groq(api_key=my_api_key)
model="llama-3.3-70b-versatile"


def llm_ans(prompt):
    sys_prompt =f"""
#ROLE:
You are a support assistant at a mobile/laptop company
#TASK
You have to classify the issue from {prompt} in a category
#CONSTRAINT
You have to classify the issue in one of three categories namely billing, technical, return.
#OUTPUT FORMAT
Your answer should be in one word only. The one word shoud be one of the categories given in constraints
#Example
For instance if a user compalin says he wants a refund then the category is Return
#FALLBACK
If the issue is unrelated to any of the categories mentioned in constraints, then the answer should be OTHER
"""
    message={
        "role":"system",
        "content": sys_prompt
    }

    user_message = {
        "role" : "user",
        "content" : prompt
    }
    messages=[message, user_message]
    response=client.chat.completions.create(model=model, messages=messages)
    ans=response.choices[0].message.content
    return ans


prompt="""
The user complaint:
I was not refunded 100 rs 
"""

print(llm_ans(prompt))