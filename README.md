# 📝 Tweet Sentiment Analysis

## 📖 Overview
This project focuses on **sentiment analysis of tweets** using **Natural Language Processing (NLP)** techniques. The dataset consists of pre-labeled tweets with sentiment labels (Positive and Negative). The project involves:
- Data Preprocessing & Cleaning
- Stopword Removal & Stemming
- Sentiment Label Mapping
- Data Visualization (Sentiment Distribution)

The goal is to **prepare the data for sentiment classification** by cleaning and standardizing text data.

---

## 📂 Dataset
The dataset used in this project is:
- **File Name:** `training.1600000.processed.noemoticon.csv`
- **Columns:**
  - `sentiment` → (0 = Negative, 4 = Positive)
  - `ids` → Unique tweet identifier
  - `date` → Timestamp of the tweet
  - `flag` → Additional metadata (not used)
  - `user` → Username of the tweet author
  - `tweet` → The actual tweet text

---

## ⚙️ Technologies Used
- **Python** (Data Analysis & NLP)
- **Pandas** (Data Manipulation)
- **NumPy** (Numerical Operations)
- **NLTK** (Text Processing & Stopwords Removal)
- **Regex (`re`)** (Text Cleaning)
- **Matplotlib & Seaborn** (Data Visualization)

---

## 📌 Project Workflow

### 🔹 1. Data Loading
- Load the dataset using **Pandas**.
- Assign column names for better readability.

### 🔹 2. Data Preprocessing
- Convert tweets to **lowercase**.
- Remove:
  - **URLs**
  - **Mentions (`@username`)**
  - **Hashtags (`#hashtag`)**
  - **Special characters & numbers**
- Remove common **stopwords** (e.g., `is`, `the`, `and`, `rt`, `amp`).
- Apply **stemming** to reduce words to their base form.

### 🔹 3. Sentiment Mapping
- Convert **numeric sentiment labels**:
  - `0` → "Negative"
  - `4` → "Positive"

### 🔹 4. Visualization
- Generate a **pie chart** to show the sentiment distribution.

---

## 📦 Installation & Setup

### 🛠 Prerequisites
Ensure you have Python installed (Python 3.7+ recommended). Install required dependencies:

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud
