# 📰 Fake News Detection using ML Techniques

## 📌 Introduction  
This project uses **Machine Learning (ML)** to classify news articles as either **real or fake**. By leveraging **Natural Language Processing (NLP)** and a **Support Vector Machine (SVM)** classifier, we analyze text-based news articles and determine their authenticity.    

---

## 📎 Loading Essential Libraries  

```python
import numpy as np
import pandas as pd
```
🔹 `numpy` - A fundamental library for numerical computations, used for handling arrays and numerical data.  
🔹 `pandas` - A library for data manipulation and analysis, used here to read and process the dataset.  

---

## 👝 Loading and Preparing the Dataset  

```python
data = pd.read_csv("fake_or_real_news.csv")
```
🔹 Reads the dataset from a CSV file into a **Pandas DataFrame**, allowing us to analyze and manipulate the data easily.  

---

## 🌂 Converting Labels (REAL → 0, FAKE → 1)  

```python
data["fake"] = data["label"].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)
```
🔹 The dataset contains a **label** column that classifies news as **REAL** or **FAKE**.  
🔹 We convert this label into numerical format:  
   - `REAL` → `0`  
   - `FAKE` → `1`  
🔹 The `apply()` function with a **lambda function** assigns `0` for real news and `1` for fake news.  
🔹 The original `label` column is **dropped** to keep only numerical data for training.  

---

## 📊 Splitting Data into Features and Labels  

```python
x, y = data["text"], data["fake"]
```
🔹 `x` (features) stores the **news articles' text**.  
🔹 `y` (labels) stores the **fake (1) or real (0) classification**.  

---

## ✂️ Splitting Data into Training & Testing Sets  

```python
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
```
🔹 The dataset is divided into two parts:  
   - **Training Set (80%)** - Used to train the model.  
   - **Testing Set (20%)** - Used to evaluate the model’s performance.  
🔹 The `train_test_split()` function ensures **random splitting** while maintaining the data distribution.  

```python
len(x_train)
len(x_test)
```
🔹 Displays the number of articles in the training and testing sets.  

---

## 💐 Text Vectorization (Converting Text to Numerical Data)  

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
```
🔹 **TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer** converts text into numerical form for the ML model.  
🔹 Removes common English **stop words** (e.g., "the", "is", "and") to improve accuracy.  
🔹 `max_df=0.7` ignores words that appear in more than **70%** of documents, reducing irrelevant features.  

```python
x_train_vectorized = vectorizer.fit_transform(x_train)
x_test_vectorized = vectorizer.transform(x_test)
```
🔹 The `fit_transform()` method:  
   - **Learns** vocabulary from `x_train`.  
   - **Transforms** `x_train` into a numerical format.  
🔹 The `transform()` method:  
   - **Applies** the learned vocabulary to `x_test`.  

---

## 🏆 Training the Fake News Classifier  

```python
from sklearn.svm import LinearSVC

classifier = LinearSVC()
classifier.fit(x_train_vectorized, y_train)
```
🔹 **Linear Support Vector Classification (LinearSVC)** is a fast and efficient **ML model** for text classification.  
🔹 The `fit()` method trains the classifier on **vectorized training data**.  

---

## 🔯 Evaluating the Model  

```python
classifier.score(x_test_vectorized, y_test)
```
🔹 Computes the model’s **accuracy** by comparing predicted labels with actual labels.  
🔹 In this case, the model achieves an accuracy of **94.08%** (`0.9408050513022889`). 🎯  

---

## 📝 Testing the Model on a Sample Article  

### 🔹 Saving a Sample Article to a File  

```python
with open("article_text.txt", "w", encoding="utf-8") as f:
    f.write(x_test.iloc[7])
```
🔹 Selects a sample news article (7th entry from `x_test`) and saves it as `article_text.txt`.  

### 🔹 Reading the Saved Article  

```python
with open("article_text.txt", "r", encoding="utf-8") as f:
    text = f.read()
```
🔹 Opens and reads the content of `article_text.txt` into the variable `text`.  

---

## 🤔 Predicting if the Article is Real or Fake  

```python
vectorized_text = vectorizer.transform([text])
classifier.predict(vectorized_text)  # -> array([1])
```
🔹 Converts the **new article** into a numerical format using `TfidfVectorizer`.  
🔹 Uses `predict()` to classify the article as **REAL (0) or FAKE (1)**.  
🔹 The output `array([1])` indicates that the model predicts this article as **FAKE**.  

---

## 🌟 Conclusion  

This project successfully detects **fake news** using **Machine Learning**! 🚀 The **Support Vector Machine (SVM)** classifier achieves an impressive **94% accuracy** in distinguishing between **real and fake** news articles.  

