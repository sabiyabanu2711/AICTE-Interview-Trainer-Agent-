# AICTE Interview Trainer Agent

> **Problem Statement #22** — An AI-powered Interview Trainer Agent built with **IBM Granite (watsonx.ai)** and **RAG (Retrieval-Augmented Generation)** that prepares candidates for job interviews through tailored question sets, model answers, and personalised preparation strategies.

---

## 🏗 Architecture

```
User (CLI or Web UI)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│              Interview Trainer Agent                     │
│                  (trainer_agent.py)                      │
└────────┬────────────────────────────┬───────────────────┘
         │                            │
         ▼                            ▼
┌─────────────────┐      ┌──────────────────────────────┐
│  Resume Parser  │      │        RAG Pipeline           │
│  (PDF/DOCX/TXT) │      │  ChromaDB  +  IBM Slate       │
│  Profile model  │      │  Embeddings (watsonx.ai)      │
└─────────────────┘      └────────────┬─────────────────┘
                                      │ Retrieved context
                                      ▼
                         ┌──────────────────────────────┐
                         │   IBM Granite LLM             │
                         │   granite-13b-chat-v2         │
                         │   granite-13b-instruct-v2     │
                         │   (watsonx.ai)                │
                         └──────────────────────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────────┐
                         │   Interview Generator         │
                         │  • Tailored questions         │
                         │  • Model answers (STAR)       │
                         │  • Improvement tips           │
                         │  • 7-day prep strategy        │
                         │  • Industry insights          │
                         └──────────────────────────────┘
```

---

## ✨ Features

| Feature | Description |
|---|---|
| **Resume Parsing** | Upload PDF / DOCX / TXT or paste text — skills, experience, education extracted automatically |
| **Tailored Questions** | IBM Granite generates role-specific questions across 4 categories: technical, behavioral, situational, HR |
| **Model Answers** | STAR-method answers for behavioral, detailed explanations for technical questions |
| **Answer Evaluation** | Submit your own answer and get a score (1–10), strengths, weaknesses, and an improved version |
| **Prep Strategy** | Personalised 7-day study plan based on your profile |
| **Industry Insights** | Current hiring trends and expectations for your target role |
| **RAG Knowledge Base** | Retrieval from a domain-specific knowledge base of interview patterns and expectations |
| **Web UI** | Clean, responsive IBM-styled browser interface |
| **CLI** | Rich interactive terminal interface for offline use |

---

## 🔧 IBM Cloud Services Used

| Service | Purpose | Tier |
|---|---|---|
| **IBM watsonx.ai** | Hosts IBM Granite LLM for question generation, answer evaluation, and strategy creation | Lite |
| **IBM Granite-13b-chat-v2** | Conversational question generation and session management | Via watsonx.ai |
| **IBM Granite-13b-instruct-v2** | Structured output: JSON question arrays, prep plans | Via watsonx.ai |
| **IBM Slate-125M Embeddings** | Semantic document embeddings for RAG retrieval | Via watsonx.ai |
| **IBM Watson Discovery** *(optional)* | External document retrieval from recruitment portals | Lite |
| **IBM Cloud Object Storage** *(optional)* | Resume storage | Lite |

---

## 🚀 Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd "AICTE-Interview Trainer Agent"
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure IBM Cloud credentials

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
IBM_CLOUD_API_KEY=your_actual_api_key
IBM_WATSON_PROJECT_ID=your_watsonx_project_id
IBM_CLOUD_URL=https://us-south.ml.cloud.ibm.com
```

#### Getting IBM Cloud credentials (Free Lite account):
1. Sign up at [ibm.com/cloud](https://cloud.ibm.com/registration)
2. Go to **watsonx.ai** → Create a project
3. Go to **Manage → Access (IAM)** → Create API key
4. Copy the Project ID from your watsonx.ai project settings

### 3. Run

**Web Interface (recommended):**
```bash
python server.py
# Open http://localhost:5000
```

**Command-Line Interface:**
```bash
python cli.py
```

---

## 📁 Project Structure

```
AICTE-Interview Trainer Agent/
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── cli.py                    # Interactive CLI entry point
├── server.py                 # Flask web server entry point
│
├── src/
│   ├── config.py             # IBM Cloud config & env loading
│   ├── granite_client.py     # IBM Granite LLM wrapper (watsonx.ai)
│   ├── rag/
│   │   └── pipeline.py       # RAG: embeddings + ChromaDB vector store
│   ├── parsers/
│   │   └── resume_parser.py  # PDF/DOCX/text resume parser
│   ├── agent/
│   │   ├── interview_generator.py  # Question + answer generation
│   │   └── trainer_agent.py        # Main orchestrator
│   └── api/
│       └── app.py            # Flask REST API
│
├── templates/
│   └── index.html            # Web UI (single-page, IBM Carbon-inspired)
│
├── data/
│   ├── knowledge_base/       # Seed JSON knowledge base files
│   │   └── interview_questions_kb.json
│   └── vector_store/         # ChromaDB persistence (auto-created)
│
├── uploads/                  # Temporary resume uploads (auto-cleared)
└── logs/                     # Application logs
```

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/prepare/profile` | Generate session from manual profile |
| `POST` | `/api/prepare/resume` | Upload resume file (multipart) |
| `POST` | `/api/prepare/text` | Paste resume as text |
| `POST` | `/api/evaluate` | Evaluate a candidate's answer |
| `POST` | `/api/knowledge` | Add custom text to knowledge base |

### Example: Generate from profile

```bash
curl -X POST http://localhost:5000/api/prepare/profile \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Priya Sharma",
    "job_title": "Data Scientist",
    "experience_level": "mid",
    "skills": ["Python", "Machine Learning", "SQL", "TensorFlow"],
    "years_of_experience": 4,
    "num_questions": 10
  }'
```

### Example: Evaluate an answer

```bash
curl -X POST http://localhost:5000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Tell me about a challenging ML project.",
    "user_answer": "I built a customer churn model using XGBoost...",
    "profile": { "job_title": "Data Scientist", "experience_level": "mid" }
  }'
```

---

## 🧑‍💻 Usage Examples

### Python API

```python
from src.agent.trainer_agent import agent

# Generate from profile
session = agent.prepare_from_profile(
    name="Alice",
    job_title="Backend Engineer",
    experience_level="senior",
    skills=["Python", "Kafka", "PostgreSQL", "AWS"],
    years_of_experience=7,
)

# Print all questions
for i, q in enumerate(session.questions, 1):
    print(f"{i}. [{q.category}] {q.question}")

# Evaluate an answer
feedback = agent.evaluate_answer(
    question="Describe a production incident you handled.",
    user_answer="We had a database outage...",
)
print(f"Score: {feedback['score']}/10")
```

---

## 📝 Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `IBM_CLOUD_API_KEY` | ✅ | — | IBM Cloud API key |
| `IBM_WATSON_PROJECT_ID` | ✅ | — | watsonx.ai project ID |
| `IBM_CLOUD_URL` | | `https://us-south.ml.cloud.ibm.com` | watsonx.ai endpoint |
| `IBM_GRANITE_MODEL_ID` | | `ibm/granite-13b-chat-v2` | Chat model |
| `IBM_GRANITE_INSTRUCT_MODEL` | | `ibm/granite-13b-instruct-v2` | Instruction model |
| `IBM_GRANITE_EMBEDDING_MODEL` | | `ibm/slate-125m-english-rtrvr` | Embedding model |
| `FLASK_PORT` | | `5000` | Web server port |
| `MAX_QUESTIONS_PER_SESSION` | | `10` | Max interview questions |
| `TOP_K_RETRIEVAL` | | `5` | RAG retrieval top-k |
| `CHUNK_SIZE` | | `512` | Document chunk size |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for AICTE Problem Statement #22 · Powered by IBM Granite on watsonx.ai*
