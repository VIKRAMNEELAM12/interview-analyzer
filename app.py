from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO

app = Flask(__name__)
nltk.download('punkt')

# Configuration
ROLE_DATA = {
    "Frontend Developer": {
        "keywords": ["react", "css", "dom", "javascript", "responsive"],
        "model": "Frontend development involves building user interfaces using HTML, CSS, and frameworks like React to create responsive web designs."
    },
    "Data Scientist": {
        "keywords": ["python", "pandas", "regression", "cleaning", "modeling"],
        "model": "Data science uses statistical models, machine learning algorithms, and data cleaning techniques to extract insights from large datasets."
    }
}

def init_db():
    conn = sqlite3.connect('database.db')
    conn.execute('CREATE TABLE IF NOT EXISTS performance (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, accuracy REAL, confidence REAL, sentiment REAL, wpm REAL)')
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_text = data.get('text', '').lower()
    role = data.get('role', 'Frontend Developer')
    duration = data.get('duration', 1)

    # 1. Similarity & Keywords
    role_info = ROLE_DATA.get(role)
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([user_text, role_info['model']])
    accuracy = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)
    
    # 2. Pacing (WPM)
    wpm = round((len(user_text.split()) / duration) * 60, 1)
    
    # 3. Sentiment & Confidence
    sentiment = round((TextBlob(user_text).sentiment.polarity + 1) * 50, 2)
    fillers = ["um", "uh", "like", "actually"]
    confidence = max(0, 100 - (sum(1 for w in user_text.split() if w in fillers) * 20))

    # 4. Keyword Matching
    user_words = set(user_text.split())
    matched = list(user_words.intersection(set(role_info['keywords'])))
    missing = list(set(role_info['keywords']) - user_words)

    # Save to DB
    conn = sqlite3.connect('database.db')
    conn.execute('INSERT INTO performance (role, accuracy, confidence, sentiment, wpm) VALUES (?, ?, ?, ?, ?)',
                 (role, accuracy, confidence, sentiment, wpm))
    conn.commit()
    conn.close()

    return jsonify({
        "accuracy": accuracy, "confidence": confidence, "sentiment": sentiment,
        "wpm": wpm, "matched": matched, "missing": missing, "role": role
    })

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, f"Interview Report: {data['role']}")
    p.drawString(100, 730, f"Accuracy: {data['accuracy']}% | Confidence: {data['confidence']}%")
    p.drawString(100, 710, f"WPM: {data['wpm']} | Sentiment: {data['sentiment']}%")
    p.drawString(100, 690, f"Keywords Found: {', '.join(data['matched'])}")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="report.pdf")

if __name__ == '__main__':
    app.run(debug=True)