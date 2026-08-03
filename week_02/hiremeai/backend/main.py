from fastapi import FastAPI
from pathlib import Path
from pypdf import PdfReader
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, Field
import os
import time
import json

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("API KEY NOT FOUND")

client = Groq(api_key=API_KEY)

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None

    skills: list[str] = []
    experiences: list[str] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()

class ChatRequest(BaseModel):
    question : str

def ask_candidate(question: str, resume: Resume):
    system_prompt = f"""
You are an AI assistant representating a job candidate.

Below is everything you know about the candidate
{resume.model_dump_json(indent=2)}

Rules:
1. Answer only using the information
2. Dont invent information
3. If information is not available
    say - "I dont have enough information"
4. Be professional
5. Answer as if HR is interviewing this candidate
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    time.sleep(2)

    return response.choices[0].message.content

def parse_resume(resume_text):
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only based on exact section headings.

    Different resumes may use different headings.

    For example:
    - Experience
    - Professional Experience
    - Work History
    - Employment
    - Internships

    These may all contain relevant experience.

    Skills may also appear in the skills section, work experience,
    internships or projects.

    Return ONLY valid JSON matching this schema:

    {resume_schema}

    Important rules:

    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """

    user_prompt = f"""
    Parse the following resume
    {resume_text}
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": 'system',
                "content": system_prompt
            },
            {
                "role" : "user",
                "content" : user_prompt
            },
        ],
        response_format={
            "type": "json_object"
        }
    )

    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)
    resume = Resume(**data)

    time.sleep(2)

    return resume


def read_pdf(filepath: Path) -> str:
    reader = PdfReader(str(filepath))
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


app = FastAPI()

@app.get("/")
def home():
    return {
        "message": "Resume extracted",
    }

@app.post("/chat")
def chat(req: ChatRequest):
    resume_path = Path(__file__).with_name("resume.pdf")
    resume_text = read_pdf(resume_path)

    resume_parsed = parse_resume(resume_text)

    answer = ask_candidate(req.question, resume_parsed)

    return {
        "Answer" : answer
    }