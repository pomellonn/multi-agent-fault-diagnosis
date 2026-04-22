## Project Structure
```
multi-agent-fault-diagnosis/
├── agent/                  Student A
│   ├── agents/
│   │   └── llm_client.py
│   ├── config/
│   │   └── agents.yaml
│   ├── main.py
│   └── agent_api.py
│
├── knowledge/              Student B
│   ├── vector_store.py
│   ├── knowledge_api.py
│   └── requirements.txt
│
├── integration/            Student C
│   ├── run_integration.py
│   ├── dataset/
│   └── requirements.txt
│
├── .env
├── .gitignore
└── requirements.txt
```



### `agent/` 

| File/Folder | What it does |
|---|---|
| `main.py` | Entry point for **manual testing**. You run this yourself to see agents talking. Not imported by anyone. |
| `agent_api.py` | Contains `send_task_to_agents()` — the function **Student C imports and calls**. This is your main deliverable. |
| `agents/llm_client.py` | Shared helper that holds the `llm_config` dict and creates `ConversableAgent` objects. Keeps repetition out of your other files. |
| `config/agents.yaml` | All agent names, system prompts, and settings in one place. When you want to tweak the Planner's behaviour, you edit this file, not the Python code. |

The relationship between these files:

```
agents.yaml  ──────────────────────────────►  llm_client.py
(defines prompts)                             (builds agents from prompts)
                                                      │
                                                      ▼
main.py  ──────────────────────────────────►  runs the conversation
agent_api.py  ─────────────────────────────►  exposes it to Student C
```

---

### `knowledge/` 

| File | What it does |
|---|---|
| `vector_store.py` | Talks to ChromaDB (the database). Saves and retrieves text as vectors. |
| `knowledge_api.py` | Exposes three functions your Retriever will call: `query_knowledge()`, `save_task_knowledge()`, `clear_knowledge_base()` |


---

### `integration/` 

| File | What it does |
|---|---|
| `run_integration.py` | The main demo script. Loads sensor data, calls your `send_task_to_agents()`, saves results, then clears the knowledge base and runs again to show improvement. |
| `dataset/` | Raw CSV files from the UMFDD dataset — healthy and bearing fault sensor readings. |

---

### How the three folders connect at runtime

```
run_integration.py (C)
        │
        │  import send_task_to_agents
        ▼
agent_api.py (A)
        │
        │  Planner → Retriever → Reasoner
        │
        │  import query_knowledge
        ▼
knowledge_api.py (B)
        │
        ▼
ChromaDB vector store
```