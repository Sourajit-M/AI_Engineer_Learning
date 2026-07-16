# take resume in pdf or word
# have hr give you a list of things like skill, experience, projects
# extract these from resume 
# match against the hr list
# generate a percentage of matching or not

import os
from pathlib import Path
from dotenv import load_dotenv
import json
from groq import Groq
from pydantic import BaseModel
import fitz

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API KEY NOT FOUND!")

client = Groq(api_key=API_KEY)

class Resume(BaseModel):
    skills: list[str]
    experience: str
    projects: list[str]

schema = Resume.model_json_schema()

system_prompt=f"""
Extract the information like skill, experience, projects from the ticket strictly based on this schema and give a json output.
{schema}
"""

system_message = {
    "role" : "system",
    "content" : system_prompt
}

def extract_pdf_text(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return text


dir = Path(__file__).resolve().parent
resume_path = dir / "resume.pdf"
text = extract_pdf_text(resume_path)

prompt=f"""
This is the resume. Please extract the information from this.
{text}
"""

user_message = {
    "role" : "user",
    "content" : prompt
}

messages = [system_message, user_message]

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    response_format={
        "type": "json_object"
    }
)

answer = response.choices[0].message.content
print(answer)

df = json.loads(answer)
resume = Resume(**df)

# print(resume.skills)
# print(resume.experience)
# print(resume.projects)

class HRRequirements(BaseModel):
    skills: list[str]
    experience: int
    projects: list[str]

hr = HRRequirements(
    skills=[
        "Python",
        "FastAPI",
        "Docker",
        "Git",
        "PostgreSQL"
    ],
    experience=2,
    projects=[
        "AI",
        "Backend"
    ]
)

user_prompt = f"""
Go through {hr} and {resume} match the skills, experience and project of the candidate and generate the percentage matching. Avoid any vague information. Only say the matched percentage.
"""

class Match(BaseModel):
    percentage: str
schema2 = Match.model_json_schema()


client2 = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content" : f"""You are helpful assistant. Go through my resume and HR requirements and give the {schema2}% match json. Do not answer anything extra"""
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],
    response_format={"type": "json_object"}
)

answer2 = client2.choices[0].message.content
df2 = json.loads(answer2)
match = Match(**df2)
print(f"Match : {match.percentage}")