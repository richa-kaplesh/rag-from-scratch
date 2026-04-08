# RAG From Scratch 

**No LangChain. No abstractions. Just pure Python, math, and curiosity.**

Most RAG tutorials hand you a 10-line LangChain snippet and call it a day.
This project builds the entire pipeline from scratch — so you actually 
understand what's happening at every step.

---

## What is RAG?

RAG (Retrieval Augmented Generation) is a technique where instead of asking 
an LLM a question directly, you first *retrieve* relevant information from 
your own documents, then pass that information along with the question to 
the LLM. This grounds the answer in your data instead of the LLM's training 
knowledge.

**The full pipeline:**
```

PDF → Extract → Chunk → Embed → Store → Query → Retrieve → LLM → Answer

```

---

## Project Structure

```

rag-from-scratch/
├── extract.py        
├── chunker.py        
├── embedder.py       
├── vector_store.py   
├── retriever.py      
├── generator.py      
├── pipeline.py       
├── main.py           
└── app.py            

```

---

## File by File — The Why Behind Every Decision

### `extract.py` — PDF Text Extraction

Extracts text from the PDF page by page, keeping track of which page each 
piece of text came from.

**Why not just extract one long string?**
If you extract everything as one big string, you lose all structural 
information. You no longer know that "the refund policy is 30 days" came 
from page 4. By extracting page by page, every chunk we create later carries 
a page number — so when the LLM answers, it can say "according to page 4" 
and the user can verify it.

**What else could we extract?**
Beyond plain text, PDFs can contain tables, images, metadata (author, 
creation date), headers/footers, and font information. For this project we 
extract plain text only. Libraries like `pdfplumber` handle tables better, 
`pytesseract` handles scanned PDFs via OCR.

---

### `chunker.py` — Splitting Text Into Chunks

Splits each page's text into smaller overlapping chunks of fixed size.

**Why chunk at all?**
If you embed an entire document as one vector, that vector becomes an 
"average" of everything — refund policy, shipping info, company history, 
all blended together. When you search for "refund policy", the match against 
this average vector is weak. Smaller chunks = more focused embeddings = 
better retrieval.

**Why overlap?**
Without overlap, a sentence sitting at the boundary of two chunks gets split. 
Neither chunk has the full context. Example:

```

Chunk 1: "...Priya is a nice girl"
Chunk 2: "but only from the outside, not inside..."

```

Neither chunk makes complete sense alone. Overlap ensures boundary sentences 
appear fully in at least one chunk.

**Why 500 characters and 50 overlap specifically?**
There's no magic number. It's a tradeoff:
- Too large → vectors are polluted with mixed topics
- Too small → vectors lack context to be meaningful

500/50 is a reasonable starting point. In production RAG systems, this is 
one of the most tuned hyperparameters. We'll experiment with different values 
in future projects and measure the impact.

**Is chunking the only option?**
No. Alternatives include:
- **Sentence-based chunking** — split on punctuation boundaries
- **Recursive chunking** — LangChain's approach, tries paragraphs first then sentences
- **Semantic chunking** — uses embeddings to find natural topic boundaries
- **Parent-document retrieval** — store small chunks for retrieval but pass larger parent chunks to the LLM

We'll implement all of these in the next project and compare results.

---

### `embedder.py` — Converting Text to Vectors

Converts chunks and queries into numerical vectors using a transformer model.

**Why do we need embeddings?**
Computers can't compare meaning in text directly. By converting text to 
vectors, we move into a mathematical space where "cheap" and "affordable" 
end up pointing in similar directions — even though the words are completely 
different. This is what makes semantic search possible.

**Why `all-MiniLM-L6-v2`?**
It's small (~90MB), fast, runs on CPU, and produces 384-dimensional vectors 
that are good enough for most RAG use cases. It's a great starting point.

**What else can you use?**
- `text-embedding-3-small` — OpenAI's embedding model, better quality, costs money
- `BGE-M3` — strong open source option, multilingual
- `Cohere Embed` — another paid option with strong performance
- Any HuggingFace sentence-transformer model

The beauty of isolating embedding in its own file — swapping models is a 
one-line change.

**How does embedding actually happen?**
The model tokenizes your text, passes it through multiple transformer layers, 
and outputs a fixed-size vector (384 numbers for this model) that represents 
the semantic meaning of the entire input. Similar meanings → vectors pointing 
in similar directions.

---

### `vector_store.py` — Cosine Similarity From Scratch

Stores chunk embeddings and finds the most similar ones for a given query.

**Why cosine similarity?**
We care about the *direction* of vectors, not their *magnitude*. A short 
sentence and a long sentence about the same topic should score as similar. 
Cosine similarity measures the angle between vectors — magnitude doesn't 
affect it. Euclidean distance would penalize length differences unfairly.

**The math:**
```

cosine_similarity = (A · B) / (||A|| × ||B||)

```
Dot product divided by the product of magnitudes. Pure numpy, no libraries.

**What else could we use?**
- **Dot product** — faster but magnitude-sensitive
- **Euclidean distance** — good for low dimensions, not ideal for high-dimensional embeddings
- **FAISS** — Facebook's library, same cosine similarity but optimized for millions of vectors
- **Approximate Nearest Neighbor (ANN)** — trades tiny accuracy loss for massive speed gains at scale

**Why top-k=3?**
You need enough chunks to cover the answer, but not so many that you flood 
the LLM's context window with irrelevant text. 3 is a starting default. 
If an answer spans multiple sections of a document, k=5 or k=7 might work 
better. This is another hyperparameter we'll tune with an eval harness in 
future projects.

---

### `retriever.py` — Finding Relevant Chunks

Takes the user's query, embeds it, and returns the top-k most relevant chunks.

The query and chunks now live in the same vector space — that's what makes 
comparison meaningful. "What is the refund policy?" as a vector will be 
close to the chunk that talks about refunds, even if the exact words differ.

---

### `generator.py` — Generating the Answer

Sends the retrieved chunks + original query to an LLM and returns the answer.

**What is the Groq API key for?**
Groq is a platform that hosts open source LLMs (like Llama 3.1) and serves 
them via API — for free at reasonable rate limits. The API key identifies 
your account. Without it you can't make calls to their servers.

**Why do we need a prompt? Why not just send the question?**
The LLM has its own vast training knowledge. Without instructions, it will 
answer from that knowledge rather than your document. The prompt explicitly 
tells it: *use only the context below, nothing else*. This is what makes RAG 
answers grounded and verifiable rather than hallucinated.

**Why does the prompt include page numbers?**
So the LLM can reference them in its answer — "according to page 4..." — 
giving the user a way to verify the source.

**What is prompt injection?**
If a malicious PDF contains text like "ignore all previous instructions and 
do X", the LLM might follow it — because to the model, it's all just tokens. 
There's no structural separation between "instructions" and "data". This is 
an unsolved problem in production RAG systems.

---

### `pipeline.py` — Connecting Everything

Runs the full pipeline end to end: extract → chunk → embed → retrieve → generate.

**Does it re-parse the document on every question?**
Yes, in this version. Every request re-extracts, re-chunks, and re-embeds 
the entire document. This is fine for learning but inefficient for real use.

**How would you make it efficient?**
Separate the pipeline into two phases:
- **Indexing** (done once): extract → chunk → embed → store
- **Querying** (done per question): embed query → retrieve → generate

Store the embeddings in a persistent vector database so you never re-process 
the same document. We implement this in the Research Assistant project.

---

### `main.py` — FastAPI Backend

Exposes a `/ask` endpoint that accepts a PDF file and a query, runs the 
pipeline, and returns the answer.

### `app.py` — Streamlit Frontend

A simple UI to upload PDFs and ask questions without touching the terminal.

---

## Where is the data stored?

Right now — **in memory, temporarily**. Chunks and embeddings exist only 
during a single request. When the request ends, they're gone.

For persistent storage, production RAG systems use vector databases:
- **FAISS** — Facebook's library, local, fast, no server needed
- **ChromaDB** — local or hosted, easy to use, good for prototyping
- **Qdrant** — production-grade, supports hybrid search
- **Pinecone** — fully managed cloud vector DB
- **Weaviate** — open source, supports multimodal data

We use ChromaDB/Qdrant in the next project.

---

## What if I want Conversational RAG?

This project answers one question at a time. For a conversation, you'd 
maintain a chat history and include previous exchanges in every prompt:

```

Context: {retrieved chunks}

Conversation so far:
User: what is the notice period?
Assistant: The notice period is 30 days.
User: what about probation?

```

The LLM sees the full history and answers in context. We implement this 
in the Research Assistant project.

---

## What if the Document is Very Large?

Large documents mean more chunks, more embeddings, slower retrieval. 
Solutions:
- Use FAISS with approximate nearest neighbor search instead of brute-force cosine
- Pre-index documents and store embeddings persistently
- Use a faster embedding model
- Filter by metadata (page range, section) before semantic search

---

## How Do You Measure RAG Accuracy?

With an eval harness — a golden dataset of question/expected answer pairs. 
You run your pipeline on each question, compare the returned answer to the 
expected answer using cosine similarity or an LLM judge, and get a score.

This lets you compare techniques objectively:
*"Chunk size 200 scores 0.82, chunk size 500 scores 0.74 — smaller chunks 
win for this dataset."*

We build a full eval harness in the Research Assistant project.

---

## Stack

| Component | Tool |
|-----------|------|
| PDF Parsing | PyMuPDF |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Search | NumPy (cosine similarity from scratch) |
| LLM | Llama 3.1 8B via Groq (free) |
| Backend | FastAPI |
| Frontend | Streamlit |

---

## Run It

```bash
git clone https://github.com/yourusername/rag-from-scratch
cd rag-from-scratch
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your Groq API key to `.env`:
```
GROQ_API_KEY=your_key_here
```

Run in two separate terminals:
```bash
uvicorn main:app --reload
streamlit run app.py
```

---

## What's Next

This project covers the foundation. The next project — Research Assistant — 
builds on this with real vector databases, hybrid search, reranking, query 
expansion, HyDE, and a proper eval harness to measure everything.

---

## Intended Audience

This is not a quickstart. If you want RAG running in 10 minutes, use LangChain.

This is for developers who want to know *why* every line exists.
```

