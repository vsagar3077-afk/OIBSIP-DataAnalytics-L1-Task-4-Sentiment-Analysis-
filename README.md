# OIBSIP-DataAnalytics-L1-Task-4-Sentiment-Analysis-
The descriptive analysis provides an overview of customer purchasing behaviour. Customers can be compared based on their purchase frequency, spending amount, recency, and lifetime value. These behavioural characteristics can then be used as input features for customer segmentation using K-Means clustering.
code:
import sys
import subprocess
import os
import re
import string

# Install required packages in the current Jupyter Python environment
packages = [
    "pandas",
    "numpy",
    "matplotlib",
    "scikit-learn",
    "wordcloud",
    "nltk"
]

subprocess.check_call([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q"
] + packages)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from wordcloud import WordCloud

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))


# Find the CSV file
if os.path.exists("twitter_training.csv"):
    file_name = "twitter_training.csv"

elif os.path.exists("twitter_training(1).csv"):
    file_name = "twitter_training(1).csv"

else:
    files = [
        file for file in os.listdir()
        if file.lower().endswith(".csv")
    ]

    if len(files) > 0:
        file_name = files[0]
    else:
        raise FileNotFoundError(
            "CSV file not found. Put twitter_training.csv in the same folder as this notebook."
        )


# Load dataset
df = pd.read_csv(
    file_name,
    header=None
)

print("=" * 60)
print("DATASET")
print("=" * 60)

print("File:", file_name)
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# Keep first four columns
df = df.iloc[:, :4]

df.columns = [
    "ID",
    "Topic",
    "Sentiment",
    "Tweet"
]


# Remove missing values
df = df.dropna(
    subset=["Sentiment", "Tweet"]
)

df["Sentiment"] = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
)


print("\n" + "=" * 60)
print("SENTIMENT DISTRIBUTION")
print("=" * 60)

print(
    df["Sentiment"].value_counts()
)


# Sentiment distribution chart
plt.figure(figsize=(8, 5))

df["Sentiment"].value_counts().plot(
    kind="bar"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Text preprocessing
def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    text = re.sub(
        r"@\w+",
        "",
        text
    )

    text = re.sub(
        r"#\w+",
        "",
        text
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)


df["Clean_Tweet"] = df["Tweet"].apply(
    clean_text
)


df = df[
    df["Clean_Tweet"].str.strip() != ""
]


# TF-IDF
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = tfidf.fit_transform(
    df["Clean_Tweet"]
)

y = df["Sentiment"]


print("\n" + "=" * 60)
print("TF-IDF")
print("=" * 60)

print("TF-IDF Shape:", X.shape)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# Naive Bayes
nb_model = MultinomialNB()

nb_model.fit(
    X_train,
    y_train
)

nb_pred = nb_model.predict(
    X_test
)


# Logistic Regression
lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)


# Evaluation function
def evaluate_model(name, actual, predicted):

    accuracy = accuracy_score(
        actual,
        predicted
    )

    precision = precision_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predicted,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            actual,
            predicted,
            zero_division=0
        )
    )

    return accuracy, precision, recall, f1


nb_results = evaluate_model(
    "NAIVE BAYES",
    y_test,
    nb_pred
)

lr_results = evaluate_model(
    "LOGISTIC REGRESSION",
    y_test,
    lr_pred
)


# Confusion matrices
labels = sorted(
    df["Sentiment"].unique()
)


def plot_confusion_matrix(
    actual,
    predicted,
    title
):

    cm = confusion_matrix(
        actual,
        predicted,
        labels=labels
    )

    plt.figure(figsize=(7, 5))

    plt.imshow(cm)

    plt.title(title)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.colorbar()

    plt.xticks(
        range(len(labels)),
        labels,
        rotation=45
    )

    plt.yticks(
        range(len(labels)),
        labels
    )

    for i in range(len(labels)):
        for j in range(len(labels)):

            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    plt.tight_layout()
    plt.show()


plot_confusion_matrix(
    y_test,
    nb_pred,
    "Naive Bayes Confusion Matrix"
)

plot_confusion_matrix(
    y_test,
    lr_pred,
    "Logistic Regression Confusion Matrix"
)


# Model comparison
results = pd.DataFrame({

    "Model": [
        "Naive Bayes",
        "Logistic Regression"
    ],

    "Accuracy": [
        nb_results[0],
        lr_results[0]
    ],

    "Precision": [
        nb_results[1],
        lr_results[1]
    ],

    "Recall": [
        nb_results[2],
        lr_results[2]
    ],

    "F1 Score": [
        nb_results[3],
        lr_results[3]
    ]
})


print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.round(4)
)


results.set_index(
    "Model"
).plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# WordCloud for each sentiment
for sentiment in labels:

    text = " ".join(
        df[
            df["Sentiment"] == sentiment
        ]["Clean_Tweet"]
    )

    if text.strip():

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(text)

        plt.figure(figsize=(10, 5))

        plt.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        plt.axis("off")

        plt.title(
            sentiment + " WordCloud"
        )

        plt.show()


# Error analysis using Logistic Regression
error_df = pd.DataFrame({

    "Tweet": df.loc[
        y_test.index,
        "Tweet"
    ].values,

    "Actual": y_test.values,

    "Predicted": lr_pred
})


errors = error_df[
    error_df["Actual"] !=
    error_df["Predicted"]
]


print("\n" + "=" * 60)
print("5 MISCLASSIFIED EXAMPLES")
print("=" * 60)

if len(errors) > 0:

    print(
        errors.head(5).to_string(
            index=False
        )
    )

else:

    print("No misclassified examples found.")


print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print("""
Misclassification can occur because:

1. Tweets may contain sarcasm.
2. Twitter contains slang and informal language.
3. A tweet may contain both positive and negative words.
4. Very short tweets may not contain enough information.
5. Some tweets require additional context.
""")


# Find best model
if lr_results[3] > nb_results[3]:

    best_model = "Logistic Regression"
    best_score = lr_results[3]

else:

    best_model = "Naive Bayes"
    best_score = nb_results[3]


print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)

print(
    "Best Model:",
    best_model
)

print(
    "Best F1 Score:",
    round(best_score, 4)
)

print("""
Real-world applications:

- Social media sentiment monitoring
- Customer feedback analysis
- Brand reputation monitoring
- Product review analysis
- Customer satisfaction analysis
- Opinion analysis
""")
