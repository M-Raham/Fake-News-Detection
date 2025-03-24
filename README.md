# Fake News Detection

## 📌 Project Overview

This project focuses on detecting fake news articles using machine learning techniques. The model is trained to classify news articles as **Fake** or **Real** based on their textual content. It utilizes **Natural Language Processing (NLP)** and **Machine Learning** to analyze and predict the authenticity of news.

## 🚀 Features

- Preprocessed dataset for training
- TF-IDF vectorization for text representation
- Machine learning model for classification
- Streamlit-based web application for user-friendly interaction

## 🛠️ Technologies Used

- **Python** (Core programming language)
- **Streamlit** (Frontend UI for predictions)
- **Scikit-learn** (Machine learning)
- **Pandas, NumPy** (Data processing)
- **NLTK, re** (Text preprocessing)
- **Joblib** (Model persistence)

## 🔧 Installation & Setup

### 1️⃣ Clone the repository:

```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

### 2️⃣ Create a virtual environment & install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3️⃣ Train the model (Optional if pre-trained model exists):

```bash
python fake_news.py
```

### 4️⃣ Run the Streamlit application:

```bash
streamlit run app.py
```

- The UI will be available at: **[http://localhost:8501](http://localhost:8501)**

## 📊 Dataset

- The model is trained on the **Fake News Dataset** available from Kaggle.
- Preprocessing includes **removing stop words, lemmatization, and TF-IDF vectorization**.

## 📌 Future Enhancements

- Deep learning model implementation (LSTMs, BERT)
- Multilingual fake news detection
- Improved dataset with real-time news scraping

## 🤝 Contributing

Pull requests are welcome! Feel free to fork the repo and improve the project.

---

### ⭐ If you found this project useful, please give it a star on GitHub! ⭐

