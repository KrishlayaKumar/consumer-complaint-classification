# consumer-complaint-classification
The objective of this task is to implement a multi-class text classification model to categorize consumer complaints from the Consumer Complaint Database.
# README — CFPB Consumer Complaints: Multiclass Text Classification

**Project goal**

Build a multi-class text classification pipeline that reads complaint narratives from the CFPB Consumer Complaint Database and assigns each complaint to one of four product categories:

| Code | Category                           |
| ---- | ---------------------------------- |
| 0    | Credit reporting, repair, or other |
| 1    | Debt collection                    |
| 2    | Consumer Loan                      |
| 3    | Mortgage                           |

This repository/notebook implements the full six-stage data-science workflow described in the assessment: EDA & feature engineering, text preprocessing, model selection, model comparison, model evaluation, and prediction. The pipeline is split into sections so you can run each step on a large dataset without re-running expensive steps.

---

## Table of contents

1. Overview
2. Dataset & columns
3. Environment & installation
4. File locations & important variables
5. Quick start — run sections step-by-step
6. Section-by-section explanation (detailed)
7. Model saving & loading
8. Exporting predictions
9. Tips for working with large datasets
10. Evaluation metrics and how to interpret them
11. Troubleshooting
12. Reproducibility & versions
13. License / contact

---

## 1 — Overview

This README explains every major step performed by the notebook/script:

* How the data is loaded and filtered to the four target product categories
* How the narrative text is cleaned and tokenized
* How features are created using TF-IDF
* Which algorithms are trained and how they are compared
* How the best model is chosen, saved, and used for prediction

All code sections are designed to be run in order. If the dataset is very large, you can run only the preprocessing once and save the cleaned data to disk to avoid repeating heavy operations.

---

## 2 — Dataset & columns

You said your dataset has these columns (exact names):

```
Date received, Product, Sub-product, Issue, Sub-issue,
Consumer complaint narrative, Company public response, Company,
State, ZIP code, Tags, Consumer consent provided?, Submitted via,
Date sent to company, Company response to consumer, Timely response?,
Consumer disputed?, Complaint ID
```

Important columns used by the pipeline:

* **Consumer complaint narrative** — the complaint text (input feature)
* **Product** — the product name (label). We map the product values into the 4 numeric classes 0..3 explained above

**Example dataset path used in the notebook**: `D:\complaints.csv\complaints.csv` (Windows path). Update `CSV_PATH` if necessary.

---

## 3 — Environment & installation

Recommended: use a Python 3.8+ virtual environment.

Install required packages (example):

```bash
python -m venv venv
# activate venv (Windows)
venv\Scripts\activate
# or (mac / linux)
source venv/bin/activate

pip install --upgrade pip
pip install pandas numpy scikit-learn nltk spacy joblib matplotlib seaborn tqdm
python -m spacy download en_core_web_sm
```

Also download NLTK data (run once in Python):

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## 4 — File locations & important variables (change if needed)

* `CSV_PATH` — path to your CFPB CSV file. Example: `D:\complaints.csv\complaints.csv`
* `OUTPUT_DIR` — where intermediate artifacts (TF-IDF vectorizer, saved model, plots, sample predictions) are stored. Default: `cfpb_outputs/`
* `SAMPLE_FRAC` — set < 1.0 during development to work on a subset. For final run set to 1.0.

---

## 5 — Quick start — run sections step-by-step

Run the notebook/script in this order. Each numbered section corresponds to a code block in the notebook.

1. **Setup & imports** — install packages and import dependencies (pandas, sklearn, nltk, spacy, joblib, matplotlib).
2. **Load CSV** — read `CSV_PATH` using `pd.read_csv()` (use `low_memory=False`). If the file is huge, read `usecols` or chunk with `chunksize=`.
3. **Select columns** — keep only `['Product', 'Consumer complaint narrative']`.
4. **Map products to 4 classes** — apply mapping to `Product` values and drop rows not mapped to these 4 classes.
5. **Filter short/missing narratives** — drop nulls and remove narratives with very few characters.
6. **Remove scrubbed tokens** — remove `XXXX` and other PII placeholders.
7. **Basic cleaning** — lowercase, remove punctuation and digits.
8. **Tokenize / stopword removal / lemmatize** — (optional: use spaCy for lemmatization with `nlp.pipe` for speed).
9. **Feature extraction** — TF-IDF vectorizer (unigrams + bigrams). Save vectorizer after fitting.
10. **Train/test split** — stratified split (e.g. 80/20) using `train_test_split(..., stratify=y)`.
11. **Train models** — try Logistic Regression, MultinomialNB, LinearSVC (class_weight='balanced' if required).
12. **Compare** — compute Accuracy, Precision, Recall, F1 (macro), and show confusion matrix.
13. **Select best model** — pick best model (by macro-F1 or whichever metric you choose) and save it with `joblib.dump()`.
14. **Predict on sample text** — load vectorizer and model to run inference on new complaints.
15. **Export predictions (optional)** — to CSV, Excel, or Google Sheets.

---

## 6 — Section-by-section explanation (detailed)

Below is a short explanation of what each part does and why.

### Section 1 — Setup & Imports

* Purpose: import required libraries, download NLTK & spaCy assets if needed.
* Why: NLP libraries require initial data downloads (`punkt`, `stopwords`, `en_core_web_sm`).

**Code snippet**:

```python
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
# ... other imports
```

### Section 2 — Load CSV

* Purpose: read the CFPB CSV file into a pandas DataFrame.
* Tip: for very large files use `chunksize` and process chunk-by-chunk, or use `usecols=['Product','Consumer complaint narrative']` to reduce memory.

**Code snippet**:

```python
df = pd.read_csv(CSV_PATH, low_memory=False, usecols=['Product','Consumer complaint narrative'])
```

### Section 3 — Filter & Map to 4 Classes

* Purpose: map many possible `Product` string values into the four numeric labels 0..3.
* Note: CFPB uses many product names. Start with keyword heuristics and inspect unmapped product values to extend mapping.

**Mapping example**:

```python
category_map = {
    'Credit reporting, credit repair services, or other personal consumer reports': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}

df = df[df['Product'].isin(category_map.keys())]
df['label'] = df['Product'].map(category_map)
```

If you see unmapped products, print `df['Product'].value_counts()` and add more rules.

### Section 4 — Remove missing/short narratives & scrub tokens

* Remove rows with missing `Consumer complaint narrative`.
* Remove narratives with less than a chosen threshold (e.g. 20 chars) — these are often too short to classify.
* Remove `XXXX` and obvious PII placeholders.

**Snippet**:

```python
df.dropna(subset=['Consumer complaint narrative'], inplace=True)
df = df[df['Consumer complaint narrative'].str.len() >= 20]
df['cleaned_raw'] = df['Consumer complaint narrative'].apply(remove_scrub_tokens)
```

### Section 5 — Basic cleaning & normalization

* Lowercase the text
* Remove punctuation, digits
* Normalize whitespace

**Snippet**:

```python
def basic_clean(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text

df['clean_basic'] = df['cleaned_raw'].apply(basic_clean)
```

### Section 6 — Tokenize, stopwords, lemmatize

* Use NLTK for tokenization + common stopwords removal
* Use spaCy `nlp.pipe` to lemmatize efficiently

**Snippet**:

```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def tokenize_remove_stopwords(text):
    toks = text.split()
    toks = [t for t in toks if t not in stop_words and len(t)>1]
    return toks

# lemmatize using spacy in a pipe
lemmas = []
for doc in nlp.pipe(preprocessed_texts, batch_size=200):
    lemmas.append([token.lemma_ for token in doc])
```

### Section 7 — Feature extraction (TF-IDF)

* TF-IDF captures word importance and is efficient for sparse text classification
* Use `ngram_range=(1,2)` to capture bigrams and set `max_features` as memory permits

**Snippet**:

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df['clean_joined'])
```

### Section 8 — Train/test split

* Use stratified sampling to keep class distribution in train/test

**Snippet**:

```python
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, stratify=y, test_size=0.2, random_state=42)
```

### Section 9 — Model training & comparison

* Try multiple algorithms: logistic regression, multinomial NB, linear SVM
* Use `class_weight='balanced'` for models that support it if classes are imbalanced
* Evaluate using classification report (precision/recall/F1) and confusion matrix

**Snippet**:

```python
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(max_iter=1000, class_weight='balanced')
clf.fit(X_train, y_train)
```

### Section 10 — Final evaluation

* Evaluate the best model on the held-out test set and generate a classification report + confusion matrix image.

**Snippet**:

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

### Section 11 — Prediction examples & mapping to category names

* Map predicted integer labels back to human-readable category names:

```python
label_names = {0: 'Credit reporting, repair, or other', 1: 'Debt collection',
               2: 'Consumer Loan', 3: 'Mortgage'}

pred = model.predict(vectorizer.transform(['sample text']))[0]
print(label_names[pred])
```

---

## 7 — Save & load the model (exact commands)

**Save**:

```python
import joblib
joblib.dump(best_model, 'final_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
```

**Load**:

```python
model = joblib.load('final_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
```

**Predict**:

```python
text = "The debt collector keeps calling me"
clean = basic_clean(remove_scrub_tokens(text))
# optionally remove stopwords / lemmatize same as training
features = vectorizer.transform([clean])
label = model.predict(features)[0]
print(label, label_names[label])
```

---

## 8 — Exporting predictions to CSV or Google Sheets

**Export to CSV**:

```python
preds = model.predict(vectorizer.transform(df['clean_joined']))
df_out = df.copy()
df_out['pred_label'] = preds
df_out.to_csv('predictions.csv', index=False)
```

**Export to Google Sheets (optional)**:

* Use `gspread` + a Google Service Account JSON credentials file
* See the notebook's `GSPREAD_CREDENTIALS_PATH` section for exact code

---

## 9 — Tips for large datasets

* **Preprocess in chunks**: use `pd.read_csv(..., chunksize=100000)` to process and clean in streaming fashion, then save cleaned chunks to disk as parquet files. Later, read the parquet files for TF-IDF fitting.
* **Incremental vectorizers**: use `HashingVectorizer` for streaming scenarios (no `fit` required) but be aware it is not invertible.
* **Reduce memory**: use `usecols` to load only necessary columns, and `dtype` hints when reading.
* **Parallelize**: use `nlp.pipe` with `batch_size` and `n_process` (spaCy v3) to lemmatize in parallel.
* **Save intermediate results**: save the cleaned `clean_joined` column to a parquet file to avoid repeating tokenization/lemmatization.
* **Sub-sampling**: for model prototyping use `SAMPLE_FRAC = 0.1` then run full dataset training after tuning.

---

## 10 — Evaluation metrics & how to interpret them

* **Accuracy**: overall proportion of correct predictions. Can be misleading with imbalanced classes.
* **Precision**: of the items predicted as class X, how many were actually X. High precision => fewer false positives.
* **Recall (Sensitivity)**: of the items that truly belong to class X, how many were detected. High recall => fewer false negatives.
* **F1-score**: harmonic mean of precision and recall. Use macro-F1 to treat each class equally.

For multi-class problems where classes are imbalanced, **macro-F1** is a recommended single-number metric.

---

## 11 — Troubleshooting (common issues)

* **MemoryError when reading CSV**: use `chunksize` or read only `usecols=['Product','Consumer complaint narrative']`.
* **spaCy too slow**: increase `batch_size`, use `nlp.pipe`, and consider `n_process` if spaCy version supports it.
* **Model accuracy is poor**: try more features (increase `max_features`), add ngrams `(1,3)`, or try embeddings (transformers, Word2Vec). Also try class weighting or sampling.
* **Many unmapped products**: print the unique product strings and add explicit mapping rules.
* **Predicted labels are integers and hard to read**: map them back using `label_names` dict.

---

## 12 — Reproducibility & versions

Record your package versions when performing final experiments:

```python
import sys, sklearn, pandas, numpy, spacy
print('python:', sys.version)
print('pandas:', pandas.__version__)
print('sklearn:', sklearn.__version__)
print('numpy:', numpy.__version__)
print('spacy:', spacy.__version__)
```

---

## 13 — License & contact

If you reuse this pipeline, please cite the project and the CFPB dataset (the dataset is publicly available on the CFPB website). If you want me to adapt this README to fit a GitHub `README.md` with badges, examples, and a downloadable `requirements.txt`, tell me and I will create them.

---

**That's it — the README contains a step-by-step guide and exact commands for each stage.**

If you want, I can:

* Export this README as a `README.md` file for you to download,
* Add a short examples folder with `run_sections.ipynb` and sample output images,
* Or generate a smaller quickstart (`quickstart.md`) which only contains the minimal commands to run the full pipeline.
