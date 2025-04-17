import os
import logging
from flask import Flask, request, jsonify, render_template

try:
    from search import prepare, query, close_all
except ImportError as e:
    logging.error(f"Error importing search module: {e}")
    prepare = None
    query = None
    close_all = None

# Initialize search engine safely
if prepare:
    try:
        prepare()
    except Exception as e:
        logging.error(f"Error initializing search engine: {e}")

TEMPLATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "templates"))
app = Flask(__name__, template_folder=TEMPLATE_DIR)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query_text = data.get('query', '').strip()

    if not query_text:
        return jsonify({'error': 'Query is required'}), 400

    if not query:
        return jsonify({'error': 'Search engine not initialized'}), 500

    try:
        results = query(query_text)
        return jsonify({'results': results[:5]})  # Returning Top 5 Results
    except Exception as e:
        logging.error(f"Error processing query '{query_text}': {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8080)
    finally:
        if close_all:
            close_all()
