# Module 3 — Zepto Support Assistant

## Overview

This module implements a small GenAI support assistant for Zepto using a local RAG pipeline.

The required graded baseline runs completely offline using the `MOCK_LLM` mode. No LLM API key is required.

The system uses:

- Sentence Transformers with `all-MiniLM-L6-v2` for local embeddings
- ChromaDB for vector storage and similarity retrieval
- LangGraph StateGraph for intent routing
- Pydantic for structured output validation
- FastAPI for the `/ask` API endpoint
- Docker for local containerization

---

## Project Structure

support_assistant/

- docs/
  - doc_01.txt
  - doc_02.txt
  - doc_03.txt
  - doc_04.txt
  - doc_05.txt
  - doc_06.txt
  - doc_07.txt
  - doc_08.txt
- main.py
- requirements.txt
- Dockerfile
- README.md

---

## 1. Document Corpus

The application uses eight Zepto policy documents covering:

1. Delivery Policy
2. Returns and Refunds
3. Membership Tiers
4. Order Tracking
5. Order Cancellation
6. Damaged or Missing Items
7. Gift Cards
8. Customer Support Hours

All eight documents are loaded from the `docs/` directory.

Each document is treated as one chunk because the documents are short enough for this task.

---

## 2. Ingestion and Embedding

The `load_documents()` function in `main.py` loads all `.txt` files from the `docs/` directory.

Each document is assigned a document ID such as:

- doc_01
- doc_02
- doc_03
- doc_04
- doc_05
- doc_06
- doc_07
- doc_08

The `SentenceTransformer` model `all-MiniLM-L6-v2` generates local embeddings for every document.

The embeddings are normalized before being stored.

---

## 3. ChromaDB

ChromaDB stores the embedded policy documents in the collection:

`zepto_policies`

The collection uses cosine similarity.

The ChromaDB persistent storage is located under:

`support_assistant/chroma_db`

The application indexes all eight documents when it starts.

Verified result:

Documents indexed: 8

---

## 4. LangGraph Architecture

The application uses a LangGraph `StateGraph` with the following nodes:

### classify_intent

This node determines whether the query is:

- policy_question
- general_question

In the required mock mode, classification uses the specified keyword heuristic.

Policy keywords include:

- delivery
- return
- refund
- membership
- tracking
- cancel
- gift card
- support hours

If one of these keywords is present, the query is classified as a `policy_question`.

Otherwise it is classified as a `general_question`.

### retrieve_and_answer

This node handles policy questions.

The query is embedded using `all-MiniLM-L6-v2`.

ChromaDB retrieves the top three most similar policy chunks using cosine similarity.

In mock mode, the final answer is generated deterministically using the first retrieved chunk:

`Based on the retrieved context: ...`

### direct_answer

This node handles general questions.

In mock mode it returns the fixed response:

`I can only answer questions about Zepto policies right now.`

---

## 5. Conditional Routing

The graph uses a conditional edge after `classify_intent`.

The flow is:

classify_intent
→ policy_question → retrieve_and_answer
→ general_question → direct_answer

Both branches then continue to response validation.

The routing itself does not depend on the LLM toggle.

---

## 6. MOCK_LLM

The application defaults to:

`MOCK_LLM=1`

This is the required graded mode.

In mock mode:

- Intent classification uses the keyword heuristic.
- Policy queries still use real embedding and ChromaDB retrieval.
- Final policy answers use deterministic mock generation.
- General questions use a deterministic canned response.
- No LLM API call is made.

The optional `MOCK_LLM=0` branch is present in the application for future real-LLM integration.

---

## 7. Structured Output

The final response is validated using the Pydantic model:

`SupportResponse`

It contains:

- `answer` — string
- `sources` — list of document IDs
- `confidence` — float between 0 and 1

In mock mode:

- Policy answers contain the retrieved document IDs in `sources`.
- General answers contain an empty `sources` list.
- Confidence is deterministically set to `1.0`.

---

## 8. Structured Prompt

The application contains a structured prompt template using the required:

- Role
- Context
- Task
- Format
- Length

The prompt also includes:

- An explicit negative constraint preventing unsupported answers
- A few-shot example

The negative constraint instructs the assistant not to answer using information that is not present in the retrieved context.

The prompt is intended for the optional real-LLM generation path.

---

## 9. FastAPI

The application exposes:

`POST /ask`

Request format:

`{"query": "your question"}`

Response format:

`{"answer": "...", "sources": [...], "confidence": 1.0}`

The application can be started using:

`uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000`

---

## 10. Example API Calls

The following examples were executed with `MOCK_LLM` left at its default value.

### Policy Question

Request:

`POST /ask`

Query:

`What is the delivery fee for orders below INR 149?`

Raw JSON response:

`{"answer":"Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del","sources":["doc_01","doc_05","doc_03"],"confidence":1.0}`

The top retrieved document is `doc_01`, which contains the delivery policy and the INR 149 delivery-fee threshold.

### General Question

Request:

`POST /ask`

Query:

`What is the capital of France?`

Raw JSON response:

`{"answer":"I can only answer questions about Zepto policies right now.","sources":[],"confidence":1.0}`

This query is classified as a `general_question` and therefore does not use retrieval.

---

## 11. RAG Architecture

The complete data flow is:

INGESTION
→ docs/*.txt
→ load_documents()

EMBEDDING
→ SentenceTransformer(all-MiniLM-L6-v2)
→ normalized document embeddings

STORAGE
→ ChromaDB
→ zepto_policies collection

QUERY
→ FastAPI /ask
→ LangGraph StateGraph

INTENT ROUTING
→ classify_intent

POLICY PATH
→ retrieve_and_answer
→ query embedding
→ ChromaDB top-3 retrieval
→ mock generation

GENERAL PATH
→ direct_answer
→ fixed mock response

GENERATION
→ SupportResponse
→ Pydantic validation
→ JSON response

The ingestion stage is performed by `load_documents()` and `build_index()` in `main.py`.

The embedding stage is handled by the `SentenceTransformer` model.

The vector storage and similarity retrieval are handled by the `zepto_policies` ChromaDB collection.

The retrieval stage is performed by the `retrieve_and_answer` LangGraph node.

The final generation stage is performed by `retrieve_and_answer` for policy questions and `direct_answer` for general questions.

The `MOCK_LLM` toggle affects classification/generation behavior. Embedding and ChromaDB retrieval always run locally and do not require an LLM API.

---

## 12. Docker

The project includes a Dockerfile for local containerization.

Build the image:

`docker build -t zepto-support-assistant support_assistant`

Run the container:

`docker run -p 7860:7860 zepto-support-assistant`

The container starts FastAPI using Uvicorn on port 7860.

The Dockerfile is the required graded containerization baseline.

---

## 13. Running Locally

From the project root:

`cd Zepto-Capstone-Project`

Start the API:

`uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000`

Then send a request to:

`POST http://127.0.0.1:8000/ask`

Example:

`{"query": "What is the delivery fee for orders below INR 149?"}`

---

## Conclusion

Module 3 implements a complete offline RAG support assistant using local embeddings, ChromaDB retrieval, LangGraph orchestration, deterministic mock generation, Pydantic structured output, FastAPI, and Docker.

The required mock baseline was tested successfully with both a policy query and an unrelated general query.
