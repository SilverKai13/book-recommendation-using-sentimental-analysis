import argparse
import os
from operator import itemgetter
from pprint import pprint

import pyrebase
from dotenv import load_dotenv

load_dotenv()


def get_db():
    config = {
        "apiKey": os.environ["FIREBASE_API_KEY"],
        "authDomain": os.environ["FIREBASE_AUTH_DOMAIN"],
        "databaseURL": os.environ["FIREBASE_DATABASE_URL"],
        "storageBucket": os.environ["FIREBASE_STORAGE_BUCKET"],
    }
    firebase = pyrebase.initialize_app(config)
    return firebase.database()


def aggregate_by_book(all_books, genre):
    """all_books: {book_title: {push_key: {genre, book, comment, rating, score}, ...}, ...}

    Each book can have many reviews now (extract.py pushes under a
    generated key instead of overwriting), so this averages sentiment
    score and rating per book instead of reading a single value.
    """
    aggregated = []
    for reviews in all_books.values():
        reviews = reviews.values()
        matching = [r for r in reviews if r.get("genre") == genre]
        if not matching:
            continue
        title = matching[0]["book"]
        avg_score = sum(r["score"] for r in matching) / len(matching)
        avg_rating = sum(int(r["rating"]) for r in matching) / len(matching)
        aggregated.append({
            "book": title,
            "avg_score": avg_score,
            "avg_rating": avg_rating,
            "review_count": len(matching),
        })
    return aggregated


def parse_args():
    parser = argparse.ArgumentParser(description="Rank books in a genre by rating and by sentiment.")
    parser.add_argument("--genre", help="Genre to rank")
    return parser.parse_args()


def main():
    args = parse_args()
    genre = args.genre or input("Genre: ")

    db = get_db()
    all_books = db.child("Admin").get().val() or {}
    aggregated = aggregate_by_book(all_books, genre)

    by_rating = sorted(aggregated, key=itemgetter("avg_rating"), reverse=True)
    by_score = sorted(aggregated, key=itemgetter("avg_score"), reverse=True)

    pprint("Books according to the best Ratings:")
    pprint(by_rating)
    print("\n")
    pprint("Books according to Sentiment Analysis Score:")
    pprint(by_score)


if __name__ == "__main__":
    main()
