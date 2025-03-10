import pandas as pd
import numpy as np
import nltk
import re
import string

# NLP tools
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Machine Learning tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Saving the model
import joblib

# Load the dataset
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")

# Add labels: 0 = Fake, 1 = Real
fake_df["label"] = 0
real_df["label"] = 1

# Combine both datasets
df = pd.concat([fake_df, real_df], axis=0)

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Display dataset info
print(df.head())  
print("\n✅ Dataset Loaded Successfully!")

# Download NLTK resources (only needed once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function to clean and preprocess text
def preprocess_text(text):
    if not isinstance(text, str):  # Check for non-string values
        return ""

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation, numbers, and special characters
    text = re.sub(r'\W+', ' ', text)

    # Tokenization (split into words)
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Apply lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)

# Apply preprocessing to the dataset
df["clean_text"] = df["text"].astype(str).apply(preprocess_text)

# Display processed text
print(df[["text", "clean_text"]].head())  
print("\n✅ Text Preprocessing Completed!")

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words

# Transform text data into numerical form
X = vectorizer.fit_transform(df["clean_text"])

# Target labels (Fake = 0, Real = 1)
y = df["label"]

# Print shape of TF-IDF matrix
print(f"TF-IDF Shape: {X.shape}")  # Should be (rows, 5000)
print("\n✅ TF-IDF Feature Extraction Completed!")

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Naïve Bayes classifier
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Training Completed! Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, "fake_news_model.pkl")

# Save the TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\n✅ Model Saved Successfully!")
