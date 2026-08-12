import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key=os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API KEY NOT FOUND")

client=Groq(api_key=my_api_key)

# step 1 - create the knowledge base
knowledge_base={
    "age" : " The age of Sourajit is 21 years",
    "profession" : "Profession of Sourajit is Student"
}

# step 2 retreieval
def retrieve_info(question):
    question=question.lower()
    if "age" in question:
        return knowledge_base["age"]
    elif "profession" in question:
        return knowledge_base["profession"]
    else:
        return None


def ask_llm(question):
    context=retrieve_info(question)

    sys_prompt=f"""
    Answer in one line only. Answer only based on this context. 
    Do not hallucinate. 
    Context: {context}
    """
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": question
    }
    messages=[system_message, message]
    response=client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        messages=messages,
        temperature=0,
        max_tokens=100
    )
    answer=response.choices[0].message.content
    return answer


question="what is sourajit's age?"
print(ask_llm(question))