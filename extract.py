import argparse
import os

import pyrebase
from dotenv import load_dotenv

from sentiment import score_text

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


def parse_args():
    parser = argparse.ArgumentParser(description="Submit a book review and score its sentiment.")
    parser.add_argument("--genre", help="Book genre")
    parser.add_argument("--book", help="Book title")
    parser.add_argument("--comment", help="Review text")
    parser.add_argument("--rating", type=int, choices=range(0, 6), help="Rating, 0-5")
    return parser.parse_args()


def prompt_for_review():
    genre = input("Enter Genre: ")
    book = input("Enter Book Name: ")
    comment = input("Your Comment: ")
    while True:
        raw_rating = input("Rating 0-5: ")
        try:
            rating = int(raw_rating)
        except ValueError:
            print("Rating must be a whole number.")
            continue
        if 0 <= rating <= 5:
            break
        print("Rating must be between 0 and 5.")
    return genre, book, comment, rating


def main():
    args = parse_args()
    if args.genre and args.book and args.comment is not None and args.rating is not None:
        genre, book, comment, rating = args.genre, args.book, args.comment, args.rating
    else:
        genre, book, comment, rating = prompt_for_review()

    score = score_text(comment)
    print("Your score after sentiment analysis:")
    print(score)

    data = {"book": book, "genre": genre, "comment": comment, "rating": rating, "score": score}
    db = get_db()
    # push() generates a unique key per review, so a second review for the
    # same book no longer overwrites the first (used to be
    # db.child("Admin").child(book).set(data), keyed by title alone).
    db.child("Admin").child(book).push(data)


if __name__ == "__main__":
    main()
