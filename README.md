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

