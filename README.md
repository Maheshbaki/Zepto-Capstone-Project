# Zepto Capstone Project

## Overview

This repository contains the complete Zepto Capstone Project, covering data engineering, exploratory data analysis, machine learning, and a GenAI support assistant.

## Project Structure

Zepto-Capstone-Project/

- README.md - Overall project documentation
- requirements.txt - Combined requirements for all modules
- data_pipeline/ - Module 1: Data Pipeline
- analytics/ - Module 2: EDA and Machine Learning
- support_assistant/ - Module 3: Zepto Support Assistant

---

# Module 1 - Data Pipeline

Module 1 focuses on building the data pipeline and working with the project data.

The implementation and supporting files are located in the data_pipeline directory.

---

# Module 2 - EDA and Machine Learning

Module 2 uses the Titanic dataset to demonstrate an end-to-end analytics and machine-learning workflow.

## Exploratory Data Analysis

The EDA workflow includes:

- Dataset inspection
- Missing-value analysis
- Data cleaning
- Outlier analysis
- Survival analysis
- Feature relationships
- Correlation analysis
- Data visualization
- Feature preparation

The original dataset contained 891 observations.

The final cleaned dataset contains 889 observations and 15 columns.

## Classification Models

The following models were evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Random Forest with SMOTE
- Tuned Random Forest

### Logistic Regression Results

Accuracy: 0.8258

Precision: 0.8136

Recall: 0.7059

F1 Score: 0.7559

ROC-AUC: 0.8647

Logistic Regression provided the strongest overall classification performance among the tested models.

### Random Forest with SMOTE

Accuracy: 0.8034

Precision: 0.7463

Recall: 0.7353

F1 Score: 0.7407

ROC-AUC: 0.8314

SMOTE improved recall compared with the original Random Forest.

### Tuned Random Forest

Best parameters:

- n_estimators = 200
- max_depth = 5
- min_samples_split = 2
- min_samples_leaf = 1

Best cross-validation F1 score: 0.7654

Test ROC-AUC: 0.8402

## Fare Regression

Linear Regression was used to predict passenger fare.

MAE: 18.553

RMSE: 41.5188

R2: 0.3539

Adjusted R2: 0.2376

## Model Persistence

The final Logistic Regression pipeline was saved using joblib as:

final_logistic_model.joblib

The reloaded model achieved an accuracy of 0.8258.

---

# Module 3 - Zepto Support Assistant

Module 3 implements an offline RAG-based customer support assistant.

The application uses:

- Sentence Transformers
- all-MiniLM-L6-v2
- ChromaDB
- LangGraph
- Pydantic
- FastAPI
- Docker
- MOCK_LLM mode

## Architecture

The system follows this workflow:

Zepto Policy Documents
→ Document Ingestion
→ Sentence Transformer Embeddings
→ ChromaDB
→ User Query
→ LangGraph Intent Classification
→ Retrieval or Direct Answer
→ Mock Generation
→ Pydantic Validation
→ FastAPI /ask

## Policy Documents

The support assistant contains eight policy documents covering:

1. Delivery
2. Returns and refunds
3. Membership
4. Order tracking
5. Order cancellation
6. Damaged or missing items
7. Gift cards
8. Customer support

## Retrieval

All eight documents are embedded using the local all-MiniLM-L6-v2 model.

The embeddings are stored in a ChromaDB collection named:

zepto_policies

The system retrieves the top three most similar documents for policy questions.

## LangGraph

The LangGraph StateGraph contains:

1. classify_intent
2. retrieve_and_answer
3. direct_answer
4. validate_response

Policy questions are routed to retrieval.

General questions are routed to the direct-answer node.

## MOCK_LLM

The application defaults to MOCK_LLM mode.

No external LLM API key is required.

Policy questions use deterministic mock generation based on the retrieved context.

General questions return:

I can only answer questions about Zepto policies right now.

## Structured Output

Responses are validated using Pydantic.

The response contains:

- answer
- sources
- confidence

## FastAPI

The application exposes:

POST /ask

Example policy query:

What is the delivery fee for orders below INR 149?

Example response:

answer: Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del

sources: doc_01, doc_05, doc_03

confidence: 1.0

Example general query:

What is the capital of France?

Response:

answer: I can only answer questions about Zepto policies right now.

sources: empty

confidence: 1.0

## Docker

A Dockerfile is included inside support_assistant.

The application can be containerized using:

docker build -t zepto-support-assistant support_assistant

The container exposes port 7860.

---

# Installation

Install the complete project's dependencies using:

pip install -r requirements.txt

Module 3 also contains its own requirements.txt inside support_assistant.

---

# Running Module 2

The Module 2 notebooks are located inside analytics.

Run:

1. module_2_EDA.ipynb
2. module_2_modeling.ipynb

---

# Running Module 3

From the repository root, run:

uvicorn support_assistant.main:app --host 0.0.0.0 --port 8000

Then send requests to:

POST /ask

The application defaults to MOCK_LLM mode.

---

# Technologies

## Data Engineering

- Python
- Pandas
- NumPy

## Machine Learning

- Scikit-learn
- Imbalanced-learn
- Matplotlib
- Seaborn
- Joblib

## GenAI and RAG

- Sentence Transformers
- ChromaDB
- LangGraph
- Pydantic

## API and Deployment

- FastAPI
- Uvicorn
- Docker

---

# Documentation

Module-specific documentation is available in:

- data_pipeline/
- analytics/
- support_assistant/

The root README provides the overall project description.

The module README files provide more detailed implementation information.

---

# Repository

GitHub Repository:

https://github.com/Maheshbaki/Zepto-Capstone-Project
