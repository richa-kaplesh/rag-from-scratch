from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pipeline import run_pipeline
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask")
async def ask(file: UploadFile, query: str):
    file_path = f"temp_{file.filename}"
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        answer = run_pipeline(file_path, query)
        return {"answer": answer}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)