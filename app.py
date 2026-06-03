#!/usr/bin/env python3
import os
import json
import re
import threading
import logging
from collections import Counter

import requests
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# -------------------- CONFIG --------------------
PROXY = {
    'http': 'socks5h://127.0.0.1:9050',
    'https': 'socks5h://127.0.0.1:9050'
}
BASE_URL = "https://akiral2iz6a7qgd3ayp3l6yub7xx2uep76idk3u2kollpj5z3z636bad.onion"
LEAKS_ENDPOINT = "/l?page={page}&sort=date:desc"
NEWS_ENDPOINT = "/n?page={page}&sort=date:desc"
LEAKS_FILE = "leaks_data.json"
NEWS_FILE = "news_data.json"
REQUEST_DELAY = 1
MAX_PAGES = 500
UPDATE_INTERVAL_HOURS = 0    # disable auto-updates on Render (no Tor)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "akira_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = int(os.environ.get("PORT", 5000))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("akira-server")

# -------------------- GLOBALS --------------------
vector_collection = None
embed_model = None
vector_db_ready = False
db_lock = threading.Lock()

# -------------------- DATA LOADING (fast) --------------------
def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

leaks = load_json(LEAKS_FILE)
news = load_json(NEWS_FILE)
logger.info(f"Loaded {len(leaks)} leaks, {len(news)} news")

# -------------------- HELPER --------------------
def clean_link(raw_url):
    if not raw_url:
        return None
    match = re.search(r'https?://[^\s\]\]]+', raw_url)
    if match:
        return match.group(0)
    raw_url = raw_url.strip()
    if raw_url.startswith(';'):
        raw_url = raw_url[1:]
    if ';;;' in raw_url:
        parts = raw_url.split(';;;')
        raw_url = parts[-1]
    raw_url = raw_url.replace(']download]', '').strip()
    return raw_url if raw_url.startswith('http') else None

# -------------------- VECTOR DB BUILD (background) --------------------
def build_vector_db():
    global vector_collection, embed_model, vector_db_ready
    logger.info("Starting background vector DB build...")
    try:
        chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
        except:
            pass
        collection = chroma_client.create_collection(COLLECTION_NAME)
        model = SentenceTransformer(EMBEDDING_MODEL)

        docs = []
        metadatas = []
        ids = []

        # Leaks
        for idx, item in enumerate(leaks):
            name = item.get('name', '').replace('\n', ' ').strip()
            desc = item.get('desc', '').replace('\n', ' ')
            date = item.get('date', '')
            url = clean_link(item.get('url'))
            text = f"Date: {date}\nName: {name}\nDescription: {desc}"
            if url:
                text += f"\nURL: {url}"
            docs.append(text)
            metadatas.append({"type": "leak", "date": date, "name": name, "url": url or ""})
            ids.append(f"leak_{idx}_{date}_{name[:30]}")

        # News
        for idx, item in enumerate(news):
            title = item.get('title', '').replace('\n', ' ').strip()
            content = item.get('content', '').replace('\n', ' ')
            date = item.get('date', '')
            text = f"Date: {date}\nTitle: {title}\nContent: {content}"
            docs.append(text)
            metadatas.append({"type": "news", "date": date, "title": title})
            ids.append(f"news_{idx}_{date}_{title[:30]}")

        # Batch add
        batch_size = 128
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            embeddings = model.encode(batch_docs).tolist()
            collection.add(ids=batch_ids, embeddings=embeddings, metadatas=batch_metas, documents=batch_docs)

        with db_lock:
            vector_collection = collection
            embed_model = model
            vector_db_ready = True
        logger.info(f"Vector DB ready: {collection.count()} documents")
    except Exception as e:
        logger.error(f"Vector DB build failed: {e}")

# Start background builder
threading.Thread(target=build_vector_db, daemon=True).start()

# -------------------- FLASK APP --------------------
app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    leak_dates = [l.get('date', '')[:7] for l in leaks if l.get('date')]
    news_dates = [n.get('date', '')[:7] for n in news if n.get('date')]
    return jsonify({
        "total_leaks": len(leaks),
        "total_news": len(news),
        "leaks_per_month": dict(Counter(leak_dates)),
        "news_per_month": dict(Counter(news_dates)),
        "latest_leak": leaks[0].get('date') if leaks else None,
        "latest_news": news[0].get('date') if news else None
    })

@app.route('/api/leaks')
def get_leaks():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    start = (page-1)*per_page
    end = start + per_page
    items = []
    for l in leaks[start:end]:
        items.append({
            "date": l.get('date', ''),
            "name": l.get('name', '').replace('\n', ' ').strip(),
            "description": (l.get('desc', '')[:200] + '...') if len(l.get('desc', '')) > 200 else l.get('desc', ''),
            "url": clean_link(l.get('url'))
        })
    return jsonify({
        "items": items,
        "total": len(leaks),
        "page": page,
        "per_page": per_page,
        "total_pages": (len(leaks) + per_page - 1) // per_page
    })

@app.route('/api/news')
def get_news():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    start = (page-1)*per_page
    end = start + per_page
    items = []
    for n in news[start:end]:
        items.append({
            "date": n.get('date', ''),
            "title": n.get('title', '').replace('\n', ' ').strip(),
            "content": (n.get('content', '')[:200] + '...') if len(n.get('content', '')) > 200 else n.get('content', '')
        })
    return jsonify({
        "items": items,
        "total": len(news),
        "page": page,
        "per_page": per_page,
        "total_pages": (len(news) + per_page - 1) // per_page
    })

@app.route('/api/search')
def search():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({"error": "Missing query"}), 400
    if not vector_db_ready:
        return jsonify({"error": "Vector database is still loading, please try again in a moment."}), 503
    top_k = int(request.args.get('top_k', 5))
    emb = embed_model.encode(q).tolist()
    results = vector_collection.query(query_embeddings=[emb], n_results=top_k)
    hits = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            hits.append({
                "text": doc[:300] + "...",
                "metadata": meta,
                "score": 1.0 - (results['distances'][0][i] if results['distances'] else 0)
            })
    return jsonify({"query": q, "results": hits})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({"error": "No question"}), 400
    if not vector_db_ready:
        return jsonify({"error": "Vector database is still loading, please try again in a moment."}), 503

    emb = embed_model.encode(question).tolist()
    results = vector_collection.query(query_embeddings=[emb], n_results=5)
    contexts = []
    sources = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            contexts.append(doc)
            meta = results['metadatas'][0][i]
            name = meta.get('name') or meta.get('title', '')
            sources.append(f"{meta.get('type','')} - {name} ({meta.get('date','')})")
    if not contexts:
        return jsonify({"answer": "No relevant information found.", "sources": []})

    full_context = "\n\n---\n\n".join(contexts)
    client = OpenAI(base_url=OPENROUTER_BASE, api_key=OPENROUTER_API_KEY)
    system_prompt = "You are a helpful assistant answering questions based only on the provided Akira ransomware leak documents and news. If the answer is not in the context, say so."
    user_prompt = f"Context:\n{full_context}\n\nQuestion: {question}\nAnswer based only on the above:"
    try:
        response = client.chat.completions.create(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        return jsonify({"answer": response.choices[0].message.content, "sources": sources})
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return jsonify({"error": str(e)}), 500

# -------------------- MAIN --------------------
if __name__ == "__main__":
    logger.info(f"Starting Flask on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)
