# Intelligent Multi-Class Sentiment Classifier

## Objective
Classify text into positive, negative, and neutral
sentiment categories using machine learning.

## Technologies
- Python
- Scikit-learn
- TF-IDF
- Logistic Regression

## Workflow
1. Prepare labelled text data.
2. Split the data into training and testing sets.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Evaluate predictions using macro F1-score.
6. Predict sentiment for new sentences.

## Preprocessing
The TF-IDF vectorizer converts text to lowercase,
removes English stop words, and uses unigrams
and bigrams.

## Run the project
Install dependencies:

pip install -r requirements.txt

Run:

python main.py

## Dataset limitation
This is a small demonstration dataset created for
learning. Its results do not represent performance
on real-world language data.
