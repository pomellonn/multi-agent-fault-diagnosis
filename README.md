# IACoP — Industrial Agent Collaboration Platform

> A multi-agent AI system where specialized agents (Planner, Retriever, Reasoner) collaborate to perform industrial fault diagnosis, with a knowledge base that improves accuracy over repeated runs.

---

## Overview

IACoP demonstrates a **Retrieval-Augmented Generation (RAG) feedback loop** applied to predictive maintenance. On the first run, agents diagnose a fault with no prior knowledge. After each run, the system saves what it learned. On subsequent runs, agents retrieve relevant historical cases and produce more confident, accurate diagnoses.

```
Input: sensor readings (vibration, temperature)
         │
         ▼
┌────────────────────┐
│   Planner Agent    │  breaks task into steps
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Retriever Agent   │  queries vector knowledge base
└────────┬───────────┘
         │
         ▼
┌────────────────────┐
│  Reasoner Agent    │  produces diagnosis + confidence score
└────────┬───────────┘
         │
         ▼
Output: { diagnosis, confidence, reasoning }
```




```
PROJECT STRUCTURE

agent/
├── agent_api.py                \\ Full knowledge integration
├── main.py                  
├── agents/
│   └── llm_client.py         \\ LLM integration
└── config/
    └── agents.yaml           \\ Agent prompts

knowledge_store.py            \\ Vector DB + Graph

run_integration.py            \\ Full demo script

dataset/
├── test_data.csv          
├── train_data.csv       
└── dataset_prepairing.py  

requirements.txt     
```

---

## How to Run 

```
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the complete demo
python3 run_integration.py
```

## The Project Demonstrates

### 1. Multi-Agent Collaboration
- Specialized agents working together
- Message passing coordination
- Emergent complex behavior

### 2. Retrieval-Augmented Generation (RAG)
- LLM reasoning augmented by retrieved knowledge
- Improvement through knowledge reuse
- Measurable performance gains

### 3. Vector Similarity Search
- Semantic search with embeddings
- Cosine distance for relevance
- Practical RAG implementation

### 4. Knowledge Graphs
- Entity relationships stored
- Parameter tracking
- Inference patterns

