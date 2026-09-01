# Simple Chatbot

A chatbot with a vanilla HTML/CSS/JS frontend and a FastAPI backend. Duckduckgo tool search added. RAG retrieval added with Qdrant vector database.

## Folder structure

```
chatbot-app/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app + CORS setup
│   │   ├── config.py            # Loads GOOGLE_API_KEY / GEMINI_MODEL from .env
│   │   ├── routes/
│   │   │   └── chat.py          # POST /api/chat endpoint
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic request/response models
│   │   └── services/
│   │       └── chat_service.py  # LangChain + Gemini chat logic, per-session history
            └── rag_service.py   # RAG implemented as a tool
│   ├── .env.example             # Copy to .env and add your Gemini API key
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## Set up your Gemini API key

1. Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `backend/.env.example` to `backend/.env`:
   ```bash
   cd backend
   cp .env.example .env
   ```
3. Open `.env` and paste your key:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

## Run the backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be live at http://127.0.0.1:8000, with interactive docs at
http://127.0.0.1:8000/docs.

## Run the frontend

Just open `frontend/index.html` in your browser (or serve it with any static
server, e.g. `python -m http.server` from inside the `frontend` folder).

The frontend calls `http://127.0.0.1:8000/api/chat` — update `API_URL` in
`script.js` if your backend runs elsewhere.

## How it works

- The bot uses **LangChain**'s `ChatGoogleGenerativeAI` wrapper to call
  **Google Gemini** (`gemini-2.0-flash` by default — change `GEMINI_MODEL` in
  `.env` to use a different model, e.g. `gemini-2.5-flash`).
- Each browser tab gets a random `session_id` (see `script.js`), and the
  backend keeps a short in-memory conversation history per session so Gemini
  has context across turns. History resets when the server restarts — swap
  the in-memory dict in `chat_service.py` for Redis/a database if you need
  it to persist.
- If `GOOGLE_API_KEY` is missing or invalid, the API returns a friendly error
  message instead of crashing.

## Next steps

- Restrict `allow_origins` in `main.py` to your real frontend URL before
  deploying.
- Swap the in-memory session history for persistent storage if needed.
- Add streaming responses for a more "live typing" feel.
