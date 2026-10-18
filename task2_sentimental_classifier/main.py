from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report

# Small demonstration dataset with three sentiment classes
texts = [
    "I love this product", "This is wonderful", "I am very happy",
    "Amazing quality", "I really enjoyed it", "Excellent experience",
    "This is fantastic", "I feel great", "Very pleased with it",
    "Best purchase ever", "I like it a lot", "Truly awesome",

    "I hate this product", "This is terrible", "I am very unhappy",
    "Awful quality", "I regret buying it", "Horrible experience",
    "This is disappointing", "I feel bad", "Very poor service",
    "Worst purchase ever", "I dislike it", "Truly awful",

    "The product is okay", "It is an average item",
    "Nothing special about it", "The quality is acceptable",
    "It works as expected", "An ordinary experience",
    "The product is fine", "Neither good nor bad",
    "It is somewhat average", "A standard product",
    "The experience was normal", "It is reasonable"
]

labels = (
    ["positive"] * 12
    + ["negative"] * 12
    + ["neutral"] * 12
)

# Split the data while preserving all three classes
X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.25,
    random_state=42,
    stratify=labels
)

# TF-IDF converts text into numerical features
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )),
    ("classifier", LogisticRegression(max_iter=1000))
])

# Train and evaluate the classifier
model.fit(X_train, y_train)
predictions = model.predict(X_test)

score = f1_score(
    y_test, predictions, average="macro"
)

report = classification_report(
    y_test, predictions, zero_division=0
)

print("MULTI-CLASS SENTIMENT CLASSIFIER")
print("-" * 40)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print(f"Macro F1-score: {score:.3f}")
print("\nClassification report:")
print(report)

# Test new sentences
new_texts = [
    "I absolutely love it",
    "This is a horrible experience",
    "The product is okay"
]

print("\nSample predictions:")
for sentence, sentiment in zip(
    new_texts, model.predict(new_texts)
):
    print(f"{sentence} -> {sentiment}")

# Save evaluation results
with open("results.txt", "w", encoding="utf-8") as file:
    file.write("Multi-Class Sentiment Classifier\n")
    file.write(f"Training samples: {len(X_train)}\n")
    file.write(f"Testing samples: {len(X_test)}\n")
    file.write(f"Macro F1-score: {score:.3f}\n\n")
    file.write(report)
    file.write("\nSample predictions:\n")
    for sentence, sentiment in zip(
        new_texts, model.predict(new_texts)
    ):
        file.write(f"{sentence} -> {sentiment}\n")

print("\nResults saved to results.txt")
