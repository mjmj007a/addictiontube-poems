import os
import json
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from openai import OpenAI, APIError
import tiktoken
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize OpenAI and Pinecone clients
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("addictiontube-poems")

# Load poems JSON for RAG context
json_path = r'videos_revised_with_poems-july04.json'
try:
    with open(json_path, 'r', encoding='utf-8') as f:
        poems = json.load(f)
    poem_dict = {poem["video_id"]: poem["poem"] for poem in poems if "video_id" in poem and "poem" in poem}
except FileNotFoundError:
    poem_dict = {}
    print(f"Warning: JSON file not found at {json_path}. RAG endpoint will not include poem text.")
except json.JSONDecodeError:
    poem_dict = {}
    print("Warning: Invalid JSON format in the file. RAG endpoint will not include poem text.")

def strip_html(text):
    return re.sub(r'<[^>]+>', '', text or '')

def log_debug(message):
    print(message)
    with open("poem_search_debug_20250705.log", "a", encoding="utf-8") as f:
        f.write(message + "\n")

# Health check route
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/search_poems', methods=['GET'])
def search_poems():
    query = request.args.get('q', '').strip()  # Remove leading/trailing whitespace
    category_id = request.args.get('category_id', '').strip()
    page = max(1, int(request.args.get('page', 1)))  # Ensure page >= 1
    size = max(1, int(request.args.get('per_page', 5)))  # Ensure size >= 1

    if not category_id:
        return jsonify({"error": "Missing category_id"}), 400
    if not query:
        return jsonify({"error": "Missing search query"}), 400  # Explicit error for blank query

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except APIError as e:
        return jsonify({"error": "OpenAI embedding failed", "details": str(e)}), 500

    try:
        top_k = max(100, size * page)  # Ensure enough results for pagination
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter={"category_id": {"$eq": category_id}}
        )
        matches = results.matches
        total = len(matches)
        start = (page - 1) * size
        end = min(start + size, total)
        paginated = matches[start:end] if start < total else []

        poems = []
        for m in paginated:
            poems.append({
                "id": m.id,
                "score": m.score,
                "title": m.metadata.get("title", "N/A"),
                "description": m.metadata.get("description", "")
            })
        return jsonify({"results": poems, "total": total})
    except Exception as e:
        return jsonify({"error": "Pinecone query failed", "details": str(e)}), 500

@app.route('/rag_answer_poems', methods=['GET'])
def rag_answer_poems():
    query = request.args.get('q', '')
    category_id = request.args.get('category_id', '')

    if not query or not category_id:
        return jsonify({"error": "Missing query or category_id"}), 400

    try:
        embedding_response = client.embeddings.create(
            input=query,
            model="text-embedding-ada-002"
        )
        query_embedding = embedding_response.data[0].embedding
    except APIError as e:
        return jsonify({"error": "Embedding failed", "details": str(e)}), 500

    try:
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter={"category_id": {"$eq": category_id}}
        )

        context_docs = []
        for match in results.matches:
            poem_text = poem_dict.get(match.id, match.metadata.get("description", ""))
            context_docs.append(strip_html(poem_text)[:3000])

        encoding = tiktoken.encoding_for_model("gpt-4")
        total_tokens = sum(len(encoding.encode(doc)) for doc in context_docs)
        log_debug(f"DEBUG: total_tokens = {total_tokens}")

        context_text = "\n\n---\n\n".join(context_docs)

        system_prompt = "You are an expert assistant for addiction recovery poetry."
        user_prompt = f"""Use the following recovery poems to answer the question.

{context_text}

Question: {query}
Answer:"""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        answer = response.choices[0].message.content.replace("—", ", ")
        return jsonify({"answer": answer})
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        log_debug("ERROR during OpenAI ChatCompletion:\n" + traceback_str)
        return jsonify({
            "error": "RAG processing failed",
            "details": str(e),
            "traceback": traceback_str
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)