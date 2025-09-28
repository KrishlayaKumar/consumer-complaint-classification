# Consumer Complaint Classification using NLP

## 1. Problem Statement

The objective is to train a machine learning model using Natural Language Processing (NLP) techniques to accurately classify raw consumer complaint text into one of the four required financial product categories:

| Label | Category                                                                     |
| ----- | ---------------------------------------------------------------------------- |
| 0     | Credit reporting, credit repair services, or other personal consumer reports |
| 1     | Debt collection                                                              |
| 2     | Consumer Loan                                                                |
| 3     | Mortgage                                                                     |

---

## 2. Dataset

The project utilizes the **Consumer Complaint Database** maintained by the **Consumer Financial Protection Bureau (CFPB)**.

* **Dataset Link:** [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)

### Explanation

This dataset contains millions of real-world complaints about financial products and services, including the consumer's written narrative (if consent was provided) and the product the complaint is about. It serves as labeled data for this supervised classification problem.

### Dataset Columns

The original dataset had **18 columns**, but only two are primarily used:

* **Consumer complaint narrative** → Feature (text input)
* **Product** → Target label (category)

Other columns include: `Date received, Sub-product, Issue, Sub-issue, Company, State, ZIP code, Tags, Consumer consent provided?, Submitted via, Date sent to company, Company response to consumer, Timely response?, Consumer disputed?, Complaint ID`.

---

## 3. Data Preprocessing

### 3.1 Column Selection & Cleaning

```python
df = df[['Consumer complaint narrative', 'Product']]
df = df.dropna(subset=['Consumer complaint narrative'])
```

**Why:** Keeps only the relevant feature (complaint text) and target (product). Drops rows where complaint text is missing.

---

### 3.2 Category Filtering & Mapping

```python
category_map = {
    "Credit reporting, credit repair services, or other personal consumer reports": 0,
    "Debt collection": 1,
    "Consumer Loan": 2,
    "Mortgage": 3
}

df = df[df['Product'].isin(category_map.keys())]
df['label'] = df['Product'].map(category_map)
```

**Why:** Restricts the dataset to four categories and converts them into numerical labels for ML.

---

### 3.3 Text Cleaning

```python
import re, string

def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"x{2,}", "", text)  # remove censored words like XXXX
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r'\d+', '', text)  # remove digits
    return text

df['clean_text'] = df['Consumer complaint narrative'].apply(clean_text)
```

**Why:** Standardizes the text for NLP by removing noise like case, punctuation, digits, and censored placeholders.

---

### 3.4 Stopword Removal

```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['clean_text'].apply(remove_stopwords)
```

**Why:** Removes common words (like "the", "is", "and") that don’t add value for classification.

---

### 3.5 Feature Extraction (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(df['clean_text'])
```

**Why:** Converts text into numerical vectors using **TF-IDF** with top 5000 features, making it suitable for ML models.

---

## 4 Model Train
---

### 4.1) NLTK Usage – Stopword Removal

**Why:**  We used NLTK (Natural Language Toolkit) as part of the text preprocessing pipeline to clean up the complaint narratives before feeding them into machine learning models.
```python
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['clean_text'].apply(remove_stopwords)
```

1) Stopwords are very common words (e.g., “the”, “is”, “at”, “and”) that appear frequently in text but carry little semantic meaning.
2) Removing them helps reduce noise in the dataset and makes the models focus on important keywords like “loan”, “mortgage”, “collector”.
3) This step improves the efficiency of feature extraction (TF-IDF) and helps models achieve better accuracy by not wasting attention on irrelevant words.

### 4.2) Logistic Regression, Naive Bayes (MultinomialNB), Support Vector Machine (LinearSVC)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

for name, model in models.items():
    print(f"\n{name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

```

<img width="649" height="320" alt="Image" src="https://github.com/user-attachments/assets/657a99f9-8006-485b-a7c3-51e917ad02c1" />
<img width="647" height="315" alt="Image" src="https://github.com/user-attachments/assets/823c0885-6517-47f9-aa16-525530b132a8" />
<img width="635" height="311" alt="Image" src="https://github.com/user-attachments/assets/3441dad8-2726-4b1a-96d6-6c5976d46dc9" />
 we have choose the Logistic regression as our main model

## confusion matrix.

<img width="642" height="599" alt="Image" src="https://github.com/user-attachments/assets/55e3d431-ec51-441d-95a2-2f8e852d87b9" />

## Model Testing

```python
sample_text = [
    "I am being harassed by debt collectors",
    "My mortgage application was denied unfairly"
]

sample_clean = [clean_text(remove_stopwords(text)) for text in sample_text]
sample_features = vectorizer.transform(sample_clean)
predictions = final_model.predict(sample_features)

for txt, pred in zip(sample_text, predictions):
    print(f"Complaint: {txt} -> Predicted Category: {pred}")
```
<img width="932" height="53" alt="Image" src="https://github.com/user-attachments/assets/c7384828-987d-4d00-942d-3c5f800987be" />

## Model save

```python
import joblib

# Save final model
joblib.dump(final_model, "final_model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

```
