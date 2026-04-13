# Multi-Agent Architectures with LangChain & LangGraph

A comprehensive Jupyter Notebook exploring **four distinct multi-agent architecture patterns** using [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/). Each pattern is demonstrated through a relatable school grading scenario, progressively increasing in complexity.

## 🎯 Overview

This project demonstrates how to build multi-agent systems that coordinate and communicate using LangGraph's state graph framework. The notebook walks through increasingly sophisticated agent orchestration patterns — from simple sequential pipelines to deeply nested hierarchical systems with dynamic routing.

**Domain:** A school grading system where teachers report scores, principals aggregate averages, and directors compare results across schools.

---

## 📦 Dependencies

```bash
pip install langchain-openai
pip install langchain_tavily
pip install langchain-community
pip install langchain_text_splitters
```

**Required environment variables:**
- `OPENAI_API_KEY` — Set via `.env` file or environment

**LLM used:** `openai:gpt-4o-mini` (via `init_chat_model`)

---

## 🏗️ Architecture Patterns

### 1. Sequential Architecture
> *Teacher → Principal → END*

The simplest pattern: agents execute in a fixed, linear order.

- **Teacher node** provides hardcoded student scores (Alice: 85, Bob: 92, Charlie: 78)
- **Principal node** receives scores via shared message state and uses the LLM to compute the average
- Flow: `START → Teacher → Principal → END`

**Key concepts:**
- `AgentState` with `Annotated[Sequence[BaseMessage], operator.add]` for message accumulation
- `make_system_prompt()` helper for collaborative agent instructions
- Streaming output via `app.stream()`

---

### 2. Hierarchical Architecture (Single Supervisor)
> *Principal routes to department teachers, then aggregates*

A supervisor (Principal) dynamically routes work to specialized worker agents using **conditional edges**.

Two independent school sub-graphs are built (School A and School B), each with:
- **Math, Science, and Sports teacher nodes** — each computes a subject average
- **Principal node** — acts as supervisor, checks which departments have reported
- **Routing logic** — `principal_router()` uses `Literal` type hints to determine the next node

**Key concepts:**
- `SchoolState` with custom `merge_dicts` reducer for the `averages` dict
- `add_conditional_edges()` for dynamic routing based on state
- Looping pattern: Principal → Department → Principal → ... → END
- Each school uses different score datasets

**School A scores:**
| Subject | Scores | Average |
|---------|--------|---------|
| Math    | 80, 90, 100 | 90.0 |
| Science | 70, 75, 80  | 75.0 |
| Sports  | 95, 100, 90 | 95.0 |
| **School Average** | | **86.67** |

**School B scores:**
| Subject | Scores | Average |
|---------|--------|---------|
| Math    | 80, 90, 70 | 80.0 |
| Science | 70, 60, 80 | 70.0 |
| Sports  | 95, 79, 90 | 88.0 |
| **School Average** | | **79.33** |

---

### 3. Super-Hierarchical — Supervisor of Supervisors (One-Way)
> *Both schools run in parallel, then report to a Director*

Composes the two school sub-graphs as nodes inside a higher-level Director graph.

- **School_A and School_B** run as compiled sub-graphs (parallel execution)
- **Director node** waits for both schools to finish, then determines the winner
- Flow: `START → [School_A, School_B] (parallel) → Director → END`

**Key concepts:**
- Sub-graph composition: compiled `StateGraph` instances used directly as nodes
- `DirectorState` with `school_results` dict to collect results from sub-graphs
- Parallel fan-out from START to both school nodes
- Fan-in: both schools converge on the Director

**Result:** 🏆 School_A wins with average 86.67

---

### 4. Super-Hierarchical — Supervisor of Supervisors (Two-Way with Conditional Edges)
> *Director sequentially dispatches to schools and re-evaluates*

The Director controls the flow using conditional edges, sending work to one school at a time.

- **Director node** checks which schools have reported and routes accordingly
- **`director_router()`** returns the next school or `__end__`
- Flow: `START → Director → School_A → Director → School_B → Director → END`

**Key concepts:**
- Sequential delegation with state-based routing
- `add_conditional_edges()` on the Director for dynamic dispatch
- The Director re-evaluates state after each school completes

---

### 5. Super-Hierarchical — Supervisor of Supervisors (Two-Way with `Command`)
> *Same as #4, but using LangGraph's `Command` API — no conditional edges needed*

The cleanest approach: routing logic is embedded directly in the Director node using `Command`.

- **Director node** returns `Command(goto="School_A", update=...)` to route
- No `add_conditional_edges()` required — logic and routing are unified
- Flow is identical to pattern #4, but with simpler graph construction

**Key concepts:**
- `Command` from `langgraph.types` enables inline routing
- `Command(goto=END, update=...)` terminates the graph
- Eliminates the separate router function pattern

---

## 🧩 State Management

All architectures use LangGraph's `TypedDict` state with **reducer annotations**:

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

class SchoolState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    averages: Annotated[dict, merge_dicts]
    school_results: Annotated[dict, merge_dicts]

class DirectorState(TypedDict):
    school_results: Annotated[dict, merge_dicts]
    messages: Annotated[Sequence[BaseMessage], operator.add]
```

The `merge_dicts` reducer merges new dictionary entries into the existing state, enabling incremental updates from different nodes.

---

## 🚀 Quick Start

1. Clone this repository
2. Create a `.env` file with your `OPENAI_API_KEY`
3. Install dependencies:
   ```bash
   pip install langchain-openai langchain_tavily langchain-community langchain_text_splitters python-dotenv
   ```
4. Open and run `agents_architecture.ipynb`

---

## 📂 Project Structure

```
Multi-Agent-architectures/
├── agents_architecture.ipynb   # Main notebook with all 5 patterns
└── README.md                   # This file
```

---

## 📚 Key Takeaways

| Pattern | Routing | Parallelism | Complexity | Use Case |
|---------|---------|-------------|------------|----------|
| Sequential | Fixed edges | None | ⭐ | Simple pipelines |
| Hierarchical (1 supervisor) | Conditional edges | None | ⭐⭐ | Task delegation |
| Super-Hierarchical (one-way) | Fixed edges | Parallel sub-graphs | ⭐⭐⭐ | Independent parallel work |
| Super-Hierarchical (two-way, conditional) | Conditional edges | Sequential sub-graphs | ⭐⭐⭐⭐ | Iterative orchestration |
| Super-Hierarchical (two-way, Command) | `Command` API | Sequential sub-graphs | ⭐⭐⭐ | Clean iterative orchestration |

---

## 🛠️ Technologies

- **LangChain** — LLM orchestration framework
- **LangGraph** — State machine graph framework for agent workflows
- **OpenAI GPT-4o-mini** — Language model for reasoning tasks
- **FAISS** — Vector store (imported, available for RAG extensions)
- **Tavily** — Search tool (imported, available for tool-augmented agents)

---

## 📄 License

This project is for educational purposes. See the repository for license details.
