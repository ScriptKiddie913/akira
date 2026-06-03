#!/usr/bin/env python3
"""
Akira Ransomware Data Server – All in One
-----------------------------------------
- Fetches leaks/news from .onion site via Tor (incremental)
- Saves JSON files (leaks_data.json, news_data.json)
- Builds vector DB (ChromaDB) for RAG
- Flask web dashboard: statistics, tables, vector search, AI chat (OpenRouter)
- Background thread updates data every 24 hours
"""

import os
import sys
import json
import re
import time
import threading
import logging
from datetime import datetime
from collections import Counter

import requests
import urllib3
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
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
UPDATE_INTERVAL_HOURS = 24          # how often to check for new data

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "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")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "akira_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("akira-server")

# Suppress SSL warnings (self-signed certs on onion)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ----------------------------------------------------------------------
# Field mappings for leaks and news
# ----------------------------------------------------------------------
FIELD_MAP = {
    "leaks": {
        "name_field": "name",
        "desc_field": "desc",
        "date_field": "date",
        "url_field": "url",
        "has_link": True
    },
    "news": {
        "name_field": "title",
        "desc_field": "content",
        "date_field": "date",
        "url_field": None,
        "has_link": False
    }
}

# ----------------------------------------------------------------------
# Tor session (used only when fetching)
# ----------------------------------------------------------------------
session = None

def get_tor_session():
    """Create a requests session with Tor proxy (lazy init)."""
    global session
    if session is None:
        session = requests.Session()
        session.proxies.update(PROXY)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:102.0) Gecko/20100101 Firefox/102.0',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'X-Requested-With': 'XMLHttpRequest',
            'Referer': BASE_URL + '/',
        })
    return session

def fetch_json(url):
    """Fetch JSON via Tor proxy."""
    try:
        sess = get_tor_session()
        resp = sess.get(url, verify=False, timeout=30)
        if resp.status_code == 200 and resp.text.strip():
            return resp.json()
        return None
    except Exception as e:
        logger.warning(f"fetch_json error: {e}")
        return None

# ----------------------------------------------------------------------
# Data fetching and incremental update
# ----------------------------------------------------------------------
def generate_item_id(item, item_type):
    fields = FIELD_MAP[item_type]
    date = item.get(fields['date_field'], '')
    name = item.get(fields['name_field'], '')
    url = item.get(fields['url_field'], '') if fields['url_field'] else ''
    return f"{date}|{name}|{url}"

def fetch_new_items(endpoint_template, item_type, existing_items):
    """Return list of new items (not in existing_items)."""
    new_items = []
    existing_ids = {generate_item_id(i, item_type) for i in existing_items}
    page = 1
    while page <= MAX_PAGES:
        url = endpoint_template.format(page=page)
        logger.info(f"Fetching {item_type} page {page}...")
        data = fetch_json(url)
        if not data or not isinstance(data, dict):
            break
        items = data.get('objects', [])
        if not items:
            break

        all_known = True
        for item in items:
            if generate_item_id(item, item_type) not in existing_ids:
                all_known = False
                break
        if all_known:
            logger.info(f"All {len(items)} items on page {page} already known – stopping.")
            break

        added = 0
        for item in items:
            item_id = generate_item_id(item, item_type)
            if item_id not in existing_ids:
                new_items.append(item)
                existing_ids.add(item_id)
                added += 1
        logger.info(f"Page {page}: got {len(items)} items, {added} new")
        page += 1
        time.sleep(REQUEST_DELAY)
    return new_items

def load_json(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def update_data():
    """Fetch new leaks and news, update JSON files. Return True if any new items."""
    logger.info("Starting data update...")
    leaks = load_json(LEAKS_FILE)
    news = load_json(NEWS_FILE)
    logger.info(f"Current: {len(leaks)} leaks, {len(news)} news")

    new_leaks = fetch_new_items(BASE_URL + LEAKS_ENDPOINT, "leaks", leaks)
    if new_leaks:
        leaks.extend(new_leaks)
        save_json(LEAKS_FILE, leaks)
        logger.info(f"Added {len(new_leaks)} new leaks. Total: {len(leaks)}")
    else:
        logger.info("No new leaks.")

    new_news = fetch_new_items(BASE_URL + NEWS_ENDPOINT, "news", news)
    if new_news:
        news.extend(new_news)
        save_json(NEWS_FILE, news)
        logger.info(f"Added {len(new_news)} new news. Total: {len(news)}")
    else:
        logger.info("No new news.")

    return bool(new_leaks or new_news)

# ----------------------------------------------------------------------
# Vector DB management
# ----------------------------------------------------------------------
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
    if raw_url.startswith('http'):
        return raw_url
    return None

def rebuild_vector_db():
    """Completely rebuild ChromaDB from current JSON files."""
    logger.info("Rebuilding vector database...")
    leaks = load_json(LEAKS_FILE)
    news = load_json(NEWS_FILE)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)

    docs = []
    metadatas = []
    ids = []

    # Process leaks
    for idx, item in enumerate(leaks):
        name = item.get('name', '').replace('\n', ' ').strip()
        desc = item.get('desc', '').replace('\n', ' ')
        date = item.get('date', '')
        url = clean_link(item.get('url'))
        text = f"Date: {date}\nName: {name}\nDescription: {desc}"
        if url:
            text += f"\nURL: {url}"
        docs.append(text)
        metadatas.append({
            "type": "leak",
            "date": date,
            "name": name,
            "url": url or ""
        })
        ids.append(f"leak_{idx}_{date}_{name[:30]}")

    # Process news
    for idx, item in enumerate(news):
        title = item.get('title', '').replace('\n', ' ').strip()
        content = item.get('content', '').replace('\n', ' ')
        date = item.get('date', '')
        text = f"Date: {date}\nTitle: {title}\nContent: {content}"
        docs.append(text)
        metadatas.append({
            "type": "news",
            "date": date,
            "title": title,
        })
        ids.append(f"news_{idx}_{date}_{title[:30]}")

    # Batch embed and add
    batch_size = 128
    for i in range(0, len(docs), batch_size):
        batch_docs = docs[i:i+batch_size]
        batch_metas = metadatas[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        embeddings = embed_model.encode(batch_docs).tolist()
        collection.add(
            ids=batch_ids,
            embeddings=embeddings,
            metadatas=batch_metas,
            documents=batch_docs
        )
    logger.info(f"Vector DB ready: {collection.count()} documents")
    return chroma_client, collection, embed_model

# ----------------------------------------------------------------------
# Flask app and global objects
# ----------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

# Global references to vector DB components (initialized after data load)
vector_collection = None
embed_model = None

# ----------------------------------------------------------------------
# Background updater thread
# ----------------------------------------------------------------------
def background_updater():
    """Periodically check for new data and rebuild vector DB if changes found."""
    while True:
        try:
            # Wait first interval before first run (so server starts quickly)
            time.sleep(UPDATE_INTERVAL_HOURS * 3600)
            logger.info("Background update: checking for new data...")
            if update_data():
                logger.info("New data found – rebuilding vector DB...")
                global vector_collection, embed_model
                _, vector_collection, embed_model = rebuild_vector_db()
            else:
                logger.info("No new data.")
        except Exception as e:
            logger.error(f"Background updater error: {e}")

def start_background_updater():
    thread = threading.Thread(target=background_updater, daemon=True)
    thread.start()
    logger.info(f"Background updater started – will check every {UPDATE_INTERVAL_HOURS} hours")

# ----------------------------------------------------------------------
# API endpoints
# ----------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/stats')
def stats():
    leaks = load_json(LEAKS_FILE)
    news = load_json(NEWS_FILE)
    leak_dates = [l.get('date', '')[:7] for l in leaks if l.get('date')]
    news_dates = [n.get('date', '')[:7] for n in news if n.get('date')]
    leak_months = Counter(leak_dates)
    news_months = Counter(news_dates)
    return jsonify({
        "total_leaks": len(leaks),
        "total_news": len(news),
        "leaks_per_month": dict(leak_months),
        "news_per_month": dict(news_months),
        "latest_leak": leaks[0].get('date') if leaks else None,
        "latest_news": news[0].get('date') if news else None
    })

@app.route('/api/leaks')
def get_leaks():
    leaks = load_json(LEAKS_FILE)
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    start = (page-1)*per_page
    end = start + per_page
    items = []
    for l in leaks[start:end]:
        items.append({
            "date": l.get('date', ''),
            "name": l.get('name', '').replace('\n', ' ').strip(),
            "description": l.get('desc', '')[:200] + '...' if len(l.get('desc', '')) > 200 else l.get('desc', ''),
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
    news = load_json(NEWS_FILE)
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))
    start = (page-1)*per_page
    end = start + per_page
    items = []
    for n in news[start:end]:
        items.append({
            "date": n.get('date', ''),
            "title": n.get('title', '').replace('\n', ' ').strip(),
            "content": n.get('content', '')[:200] + '...' if len(n.get('content', '')) > 200 else n.get('content', '')
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
    if vector_collection is None or embed_model is None:
        return jsonify({"error": "Vector DB not ready"}), 503
    top_k = int(request.args.get('top_k', 5))
    query_emb = embed_model.encode(q).tolist()
    results = vector_collection.query(query_embeddings=[query_emb], n_results=top_k)
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
    if vector_collection is None or embed_model is None:
        return jsonify({"error": "Vector DB not ready"}), 503

    # Retrieve context
    query_emb = embed_model.encode(question).tolist()
    results = vector_collection.query(query_embeddings=[query_emb], n_results=5)
    contexts = []
    sources = []
    if results['documents'] and results['documents'][0]:
        for i, doc in enumerate(results['documents'][0]):
            contexts.append(doc)
            meta = results['metadatas'][0][i]
            name = meta.get('name') or meta.get('title', '')
            sources.append(f"{meta.get('type','')} - {name} ({meta.get('date','')})")
    if not contexts:
        return jsonify({"answer": "No relevant information found in the dataset.", "sources": []})

    full_context = "\n\n---\n\n".join(contexts)

    # Call OpenRouter
    client = OpenAI(
        base_url=OPENROUTER_BASE,
        api_key=OPENROUTER_API_KEY,
    )
    system_prompt = """You are a helpful assistant that answers questions based only on the provided leaked documents and news from the Akira ransomware group. 
If the answer cannot be found in the context, say so clearly. Do not make up information."""
    user_prompt = f"""Context information:
{full_context}

Question: {question}
Answer based only on the above context:"""
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
        answer = response.choices[0].message.content
        return jsonify({"answer": answer, "sources": sources})
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return jsonify({"error": str(e)}), 500

# ----------------------------------------------------------------------
# Main startup
# ----------------------------------------------------------------------
def main():
    # On startup: load or create initial JSON files if missing
    if not os.path.exists(LEAKS_FILE) or not os.path.exists(NEWS_FILE):
        logger.info("JSON files missing – fetching initial data...")
        update_data()
    else:
        logger.info("Loading existing JSON files.")
        # Optionally check for updates immediately on startup (comment out if not wanted)
        # update_data()

    # Build vector DB from existing JSON
    global vector_collection, embed_model
    _, vector_collection, embed_model = rebuild_vector_db()

    # Start background updater
    start_background_updater()

    # Start Flask
    logger.info(f"Starting Flask server on {FLASK_HOST}:{FLASK_PORT}")
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, threaded=True)

if __name__ == "__main__":
    main()
