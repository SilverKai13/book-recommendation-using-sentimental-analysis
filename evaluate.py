"""Measure the lexicon scorer against NLTK's movie_reviews corpus.

Reports:
  - accuracy / precision / recall / confusion matrix for the hand-built
    lexicon scorer, thresholded at 0
  - lexicon coverage: what fraction of tokens in a review actually match
    a dictionary entry
  - the same accuracy metrics for two baselines: VADER (same lexicon
    family, bundled with NLTK) and TF-IDF + LogisticRegression (a learned
    baseline, evaluated on a held-out split since it's supervised)
  - a handful of the lexicon scorer's misclassified reviews, so the
    failures are concrete instead of just a number
"""

import argparse
import random

import nltk
from nltk.corpus import movie_reviews
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from sentiment import DICT_DIR, DictionaryTagger, Splitter, sentiment_score

RANDOM_SEED = 42


def load_corpus():
    """Returns a list of (text, label) pairs, label 1 = positive, 0 = negative."""
    data = []
    for category in movie_reviews.categories():
        label = 1 if category == 'pos' else 0
        for fileid in movie_reviews.fileids(category):
            data.append((movie_reviews.raw(fileid), label))
    return data


def evaluate_predictions(name, y_true, y_pred):
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    return {
        "name": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


def run_lexicon_scorer(data):
    splitter = Splitter()
    dicttagger = DictionaryTagger([
        f"{DICT_DIR}/positive.yml",
        f"{DICT_DIR}/negative.yml",
        f"{DICT_DIR}/inc.yml",
        f"{DICT_DIR}/dec.yml",
        f"{DICT_DIR}/inv.yml",
    ])

    y_true, y_pred, scores = [], [], []
    total_tokens = 0
    matched_tokens = 0
    misclassified = []

    for text, label in data:
        sentences = splitter.split(text)
        tagged_sentences = dicttagger.tag(sentences)
        score = sentiment_score(tagged_sentences)
        pred = 1 if score > 0 else 0

        for sentence in sentences:
            total_tokens += len(sentence)
        for tagged_sentence in tagged_sentences:
            for expression_form, _lemma, taggings in tagged_sentence:
                if taggings:
                    matched_tokens += len(expression_form.split())

        y_true.append(label)
        y_pred.append(pred)
        scores.append(score)
        if pred != label:
            misclassified.append((text, label, pred, score))

    coverage = matched_tokens / total_tokens if total_tokens else 0.0
    result = evaluate_predictions("Hand-built lexicon (this project)", y_true, y_pred)
    result["coverage"] = coverage
    result["misclassified"] = misclassified
    return result


def run_vader(data):
    sia = SentimentIntensityAnalyzer()
    y_true, y_pred = [], []
    for text, label in data:
        compound = sia.polarity_scores(text)["compound"]
        pred = 1 if compound > 0 else 0
        y_true.append(label)
        y_pred.append(pred)
    return evaluate_predictions("VADER (NLTK bundled lexicon)", y_true, y_pred)


def run_tfidf_logreg(data):
    texts = [text for text, _ in data]
    labels = [label for _, label in data]
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_SEED, stratify=labels
    )
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    result = evaluate_predictions("TF-IDF + LogisticRegression (80/20 split)", y_test, y_pred)
    return result


def print_result(result):
    print(f"\n{result['name']}")
    print(f"  accuracy:  {result['accuracy']:.3f}")
    print(f"  precision: {result['precision']:.3f}")
    print(f"  recall:    {result['recall']:.3f}")
    cm = result["confusion_matrix"]
    print("  confusion matrix (rows=actual, cols=predicted, order=[pos, neg]):")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")
    if "coverage" in result:
        print(f"  lexicon coverage: {result['coverage']:.1%} of tokens matched a dictionary entry")


def print_comparison_table(results):
    print("\n" + "=" * 60)
    print(f"{'method':<42}{'accuracy':>9}{'precision':>10}")
    print("-" * 60)
    for r in results:
        print(f"{r['name']:<42}{r['accuracy']:>9.3f}{r['precision']:>10.3f}")
    print("=" * 60)


def print_misclassified_examples(misclassified, n=3):
    print(f"\n{n} misclassified reviews from the lexicon scorer:")
    random.seed(RANDOM_SEED)
    sample = random.sample(misclassified, min(n, len(misclassified)))
    for text, label, pred, score in sample:
        label_name = "positive" if label == 1 else "negative"
        pred_name = "positive" if pred == 1 else "negative"
        snippet = " ".join(text.split())[:300]
        print(f"\n  actual={label_name} predicted={pred_name} score={score:.1f}")
        print(f"  \"{snippet}...\"")


def main():
    parser = argparse.ArgumentParser(description="Evaluate the lexicon scorer against baselines.")
    parser.add_argument("--examples", type=int, default=3, help="number of misclassified examples to print")
    args = parser.parse_args()

    nltk.download("movie_reviews", quiet=True)
    nltk.download("vader_lexicon", quiet=True)
    nltk.download("punkt", quiet=True)

    data = load_corpus()
    print(f"Loaded {len(data)} labeled reviews from nltk.corpus.movie_reviews")

    lexicon_result = run_lexicon_scorer(data)
    vader_result = run_vader(data)
    tfidf_result = run_tfidf_logreg(data)

    for r in (lexicon_result, vader_result, tfidf_result):
        print_result(r)

    print_comparison_table([lexicon_result, vader_result, tfidf_result])
    print_misclassified_examples(lexicon_result["misclassified"], n=args.examples)


if __name__ == "__main__":
    main()
