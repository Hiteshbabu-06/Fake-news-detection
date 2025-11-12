# ----------------------------------------------------------
# Fake News Detection Streamlit App - FINAL SAFE VERSION
# ----------------------------------------------------------

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st

# Download stopwords on first run
nltk.download('stopwords')

# -------------------------------
# Load Datasets with safe encoding
# -------------------------------
# If you get UnicodeDecodeError, use encoding='ISO-8859-1'

try:
    fake_df = pd.read_csv('Fake.csv', encoding='ISO-8859-1')
    true_df = pd.read_csv('True.csv', encoding='ISO-8859-1')
except Exception as e:
    st.error(f"Error loading CSVs: {e}")
    st.stop()

# Label: 0 = FAKE, 1 = REAL
fake_df['label'] = 0
true_df['label'] = 1

# Combine both into one DataFrame
df = pd.concat([fake_df, true_df]).reset_index(drop=True)

# Drop rows where 'text' is missing
df = df.dropna(subset=['text'])

# -------------------------------
# Text Cleaning Function (Safe)
# -------------------------------
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if pd.isnull(text):
        return ""  # handle NaN rows safely
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text_tokens = text.split()
    filtered_words = [w for w in text_tokens if w not in stop_words]
    return ' '.join(filtered_words)

# Clean all text safely
df['clean_text'] = df['text'].apply(clean_text)

# -------------------------------
# Feature Engineering: TF-IDF
# -------------------------------
X = df['clean_text']
y = df['label']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Logistic Regression Model
# -------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -------------------------------
# Evaluate the Model
# -------------------------------
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📰 Fake News Classifier")
st.write(
    "Paste a news article below to check if it's likely **FAKE** or **REAL**."
)

user_input = st.text_area("Your News Text:", height=300)

if st.button("Predict"):
    if len(user_input.strip()) == 0:
        st.warning("Please enter some text to classify.")
    else:
        cleaned_input = clean_text(user_input)
        input_tfidf = tfidf.transform([cleaned_input])
        prediction = model.predict(input_tfidf)[0]
        if prediction == 0:
            st.error("⚠️ This news is predicted to be **FAKE**.")
        else:
            st.success("✅ This news is predicted to be **REAL**.")
