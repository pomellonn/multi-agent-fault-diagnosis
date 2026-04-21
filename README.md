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

