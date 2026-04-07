from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def generate_response(query: str , retrieved_chunks: list[dict]):
    context = "\n\n".join([
        f"[Page {chunk['page_number']}]\n{chunk['text']}"
        for chunk in retrieved_chunks
    ])
    prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the context provided below.
    If the answer is not context, say "I don't know based on the provided document."

    Context:
    {context}

    Question: {query}

    Answer: """

    response = client.chat.completions.create(
                 model="llama3-8b-8192",
                 messages=[
                     {"role": "user", "content": prompt}]
           )
    return response.choices[0].message.content.strip()