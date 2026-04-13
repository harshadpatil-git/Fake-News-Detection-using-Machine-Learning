import re
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# ─── Same cleaning function used during training ─────────────────────
def strip_dateline(text):
    cleaned = re.sub(r'^[A-Z\s,]+\([^)]+\)\s*[-–]\s*', '', text.strip())
    return cleaned

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    news = strip_dateline(news)          # Clean before predicting
    transformed = tfidf.transform([news])
    prediction = model.predict(transformed)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)