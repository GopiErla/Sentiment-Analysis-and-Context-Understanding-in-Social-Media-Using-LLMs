
# 📝 Tweet Sentiment Analysis using ML, Deep Learning, and LLMs

## 📖 Overview

This project focuses on **Sentiment Analysis of Tweets** using a **layered approach**:
1. Traditional Machine Learning Models
2. Deep Learning with TensorFlow (MLP on GPU)
3. Transformer-based Large Language Models (LLMs) like **BERT** and **RoBERTa**

We aim to classify tweets as either **Positive** or **Negative** sentiments from the pre-labeled [Sentiment140](http://help.sentiment140.com/for-students/) dataset.

---

## 📂 Dataset

- **File:** `training.1600000.processed.noemoticon.csv`
- **Size:** 1.6 Million Tweets
- **Columns:**
  - `sentiment`: 0 = Negative, 4 = Positive
  - `ids`, `date`, `flag`, `user`, `tweet`: Metadata + tweet text

---

## ⚙️ Technologies Used

| Area                | Tools/Libraries                                     |
|---------------------|-----------------------------------------------------|
| Language            | Python 3.11                                         |
| Data Handling       | `pandas`, `numpy`                                   |
| NLP Preprocessing   | `nltk`, `re`, `CountVectorizer`, `PorterStemmer`    |
| Classical ML Models | `scikit-learn` (`LogReg`, `SVM`, `Naive Bayes`)     |
| Deep Learning       | `TensorFlow`, `Keras` (for GPU-accelerated MLP)     |
| LLMs & Transformers | `Hugging Face Transformers` (`BERT`, `RoBERTa`)     |
| Visualization       | `matplotlib`, `seaborn`                             |

---

## 📌 Project Workflow

### 🔹 1. Preprocessing
- Lowercasing, removing URLs, mentions, hashtags, digits, and special chars
- Removing stopwords (`amp`, `rt`, `lt`, etc.)
- Stemming using NLTK's `PorterStemmer`

### 🔹 2. Feature Engineering
- Calculated tweet length
- Transformed text into feature vectors using **CountVectorizer**

---

## 🤖 Models Implemented

### 🧠 A. Classical Machine Learning Models
- **Logistic Regression**
- **Naive Bayes**
- **Support Vector Machine (SVM)**

📊 **Sample Accuracy Scores:**
```
Logistic Regression: ~78%
Naive Bayes: ~76%
SVM: ~80%
```

---

### ⚡ B. Deep Learning with Keras (MLP on GPU)
Using TensorFlow and GPU-accelerated Colab runtime:
```python
Sequential([
    Dense(128, relu),
    Dropout(0.3),
    Dense(64, relu),
    Dropout(0.2),
    Dense(1, sigmoid)
])
```

📊 **Performance:**
- Trained on 500,000 tweets (CountVectorizer max_features=1000)
- **Accuracy:** ~81%
- Used GPU for acceleration with `tf.config.list_physical_devices('GPU')`

---

### 🤖 C. Transformer-based LLMs

#### ✅ BERT (`bert-base-uncased`)
- Fine-tuned on 50,000 tweets (2 epochs)
- Hugging Face `Trainer` API used
- **Test Accuracy:** ~82–83%
- **F1 Score:** 0.83 (macro)

#### ✅ RoBERTa (`roberta-base`)
- Also fine-tuned for 2 epochs
- Same tokenizer logic
- **F1 Score:** 0.83
- Slightly better recall for Positive class

---

## 🧪 Sample LLM Prediction Function
```python
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=64)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(dim=1).item()
        return "Positive" if pred == 1 else "Negative"
```

---

## 🧠 Observations & Insights

| Model        | Pros                                  | Cons                                |
|--------------|----------------------------------------|-------------------------------------|
| Logistic     | Fast, interpretable                    | Lower accuracy than DL/LLMs         |
| MLP (Keras)  | Simple, GPU-accelerated, flexible      | Requires scaling + tuning           |
| BERT         | Great contextual understanding         | Slower, needs GPU                   |
| RoBERTa      | Better generalization than BERT (in this case) | Slightly larger memory footprint    |

---

## 📦 Setup & Installation

### 📌 Install All Required Libraries
```bash
pip install pandas numpy nltk matplotlib seaborn scikit-learn tensorflow transformers datasets wordcloud
```

---

## 📈 Results Summary

| Model             | Accuracy | F1 Score | Notes                     |
|------------------|----------|----------|---------------------------|
| Logistic Reg.     | ~78%     | 0.78     | Basic but solid           |
| SVM               | ~80%     | 0.80     | Strong baseline           |
| MLP (TensorFlow)  | ~81%     | 0.81     | GPU-accelerated & robust  |
| BERT              | ~82–83%  | 0.83     | Good recall for negative  |
| RoBERTa           | ~83%     | 0.83     | Slightly more stable F1   |

---

## 📌 Future Improvements

- Add **neutral sentiment** class for 3-way classification
- Fine-tune LLMs for more epochs + hyperparameter tuning
- Explore multilingual sentiment (using XLM-R)
- Deploy as a **web app with Gradio or Streamlit**

---

## 👨‍💻 Author
Developed by Gopi (with love for LLMs and deep learning ❤️)
