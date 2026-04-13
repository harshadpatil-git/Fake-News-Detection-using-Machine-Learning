import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load data
fake = pd.read_csv('dataset/Fake.csv')
real = pd.read_csv('dataset/True.csv')

fake['label'] = 'FAKE'
real['label'] = 'REAL'

df = pd.concat([fake, real], ignore_index=True)

# ─── FIX: Strip the Reuters/AP dateline from real articles ───────────
# Matches patterns like: "WASHINGTON (Reuters) - " or "LONDON (AP) - "
def strip_dateline(text):
    if not isinstance(text, str):
        return text
    # Remove: "CITY (Source) - " at the start
    cleaned = re.sub(r'^[A-Z\s,]+\([^)]+\)\s*[-–]\s*', '', text.strip())
    return cleaned

df['text'] = df['text'].apply(strip_dateline)

# Combine title + text for richer features
df['content'] = df['title'] + ' ' + df['text']

x = df['content']
y = df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
x_train_tfidf = tfidf.fit_transform(x_train)
x_test_tfidf = tfidf.transform(x_test)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(x_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(x_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved!")