import os, sqlite3, nltk
from flask import Flask, render_template, request, jsonify, send_file, session
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.pdfgen import canvas
from io import BytesIO
from openai import OpenAI

app = Flask(__name__)
app.secret_key = 'super_secret_key'
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Role Data & Model Answers
ROLE_DATA = {
    "Frontend Developer": {"model": "Frontend engineering focuses on UI/UX using HTML, CSS, and JS frameworks like React."},
    "Data Scientist": {"model": "Data science involves statistical analysis, machine learning, and data cleaning."}
}

def init_db():
    with sqlite3.connect('database.db') as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, role TEXT, score REAL, date TIMESTAMP DEFAULT CURRENT_TIMESTAMP)')

init_db()

@app.route('/')
def index():
    session['chat_history'] = [] # Reset history on load
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    user_text = data.get('text', '')
    role = data.get('role', 'Frontend Developer')
    
    # 1. Scoring Logic
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([user_text, ROLE_DATA[role]['model']])
    accuracy = round(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100, 2)
    sentiment = round((TextBlob(user_text).sentiment.polarity + 1) * 50, 2)
    
    # 2. Adaptive Follow-up
    session['chat_history'].append({"role": "user", "content": user_text})
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": f"You are a technical interviewer for {role}."}] + session['chat_history'][-5:]
    )
    follow_up = response.choices[0].message.content
    session['chat_history'].append({"role": "assistant", "content": follow_up})

    return jsonify({"accuracy": accuracy, "sentiment": sentiment, "follow_up": follow_up})

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    data = request.get_json()
    buffer = BytesIO()
    p = canvas.Canvas(buffer)
    p.drawString(100, 800, f"Interview Report - {data['role']}")
    p.drawString(100, 780, f"Accuracy: {data['accuracy']}% | Sentiment: {data['sentiment']}%")
    p.showPage()
    p.save()
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name="Report.pdf")

if __name__ == '__main__':
    app.run(debug=True)