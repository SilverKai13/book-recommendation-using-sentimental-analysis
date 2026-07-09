# book-recommendation-using-sentimental-analysis

A small sentiment scorer for book reviews, plus a script that ranks books
in a genre by average rating and by average sentiment score. Reviews and
ratings go into Firebase.

I wrote this early in my degree, mostly to see if I could build a
sentiment analyzer from scratch instead of importing one. It works, in
the sense that it runs and produces a number. Whether that number means
anything is a separate question, and this README has an honest answer:
mostly no. See the results section.

## How the scoring works

No machine learning, no pretrained model. It's a lexicon: five YAML
files listing words and short phrases tagged `positive`, `negative`,
`inc` (intensifier, like "very"), `dec` (diminisher, like "barely"), and
`inv` (negation, like "not").

A review gets split into sentences and tokens with NLTK. Each token is
checked against the dictionaries. Positive words add 1, negative words
subtract 1. An intensifier doubles whatever comes after it, a diminisher
halves it, a negation flips the sign. All of that gets summed into one
number per review. Positive score, positive review.

`sentiment.py` has the scorer as an importable module (`score_text`).
`extract.py` is the CLI that takes a review and writes it to Firebase.
`two.py` reads everything back and ranks books within a genre.

## Results

I ran the scorer against NLTK's `movie_reviews` corpus — 2,000 reviews,
labeled positive or negative, 1,000 each. Not book reviews, but it's a
labeled sentiment dataset that's easy to get and old enough to fit the
project's era, and reviews-are-reviews for this purpose. Score
thresholded at zero: above is positive, at-or-below is negative.

| method                                     | accuracy | precision |
|---------------------------------------------|---------:|----------:|
| **This project's lexicon**                   |    0.582 |     0.565 |
| VADER (NLTK's bundled lexicon scorer)        |    0.635 |     0.598 |
| TF-IDF + Logistic Regression (80/20 split)   |    0.825 |     0.812 |

Coin flip is 0.50. My lexicon beats that by a little. VADER, which is
also just a lexicon but a properly built one, beats mine by five points.
TF-IDF with a plain logistic regression classifier — no deep learning,
nothing fancy, and it wasn't even close — beats both by a wide margin.
Reproduce it with `python evaluate.py`.

The bigger number, and the one I actually wanted, is **lexicon
coverage: 1.2%**. Only 1.2% of the tokens in a typical review match any
entry in my dictionaries. The other 98.8% of every review is invisible
to this scorer. It's not that the scoring logic is wrong — there just
isn't enough dictionary to work with. 30 positive words and 54 negative
words was never going to cover how people actually write about movies
or books. VADER's lexicon has about 7,500 entries. That gap is most of
the difference in the table above.

### Where it fails

Pulled straight from the confusion matrix, and traced through the
scorer to see which tokens actually moved the number:

> "stalked does not provide much suspense, though that is what it sets
> out to do..." (a ~40-sentence review) — actual: negative, predicted:
> positive (score 1.0)

"suspense" isn't in either dictionary — I checked, and my first
assumption about why this one failed was wrong. What actually happens:
of ~500 words in the review, only five ever match anything: "love",
"contrary", two separate "not ... bad" constructions, and one bare
"good", scattered across unrelated sentences. One of the "not bad"s
correctly scores negative; the other scores positive, because enough
unmatched filler words sit between "not" and "bad" that the negation
window (3 tokens) expires before it reaches it. Five essentially random
sign flips is not a signal, it's noise that happens to sum to a number.
This is the coverage problem, not the negation problem.

> "...isn't nearly as dull as this" — actual: negative, predicted:
> positive (score 2.0)

Same shape of failure. "dull" isn't in either dictionary, so the
negation logic never even engages here — "isn't nearly as dull" doesn't
touch the score at all. What actually produces the +2 is "interesting"
and "too good" showing up elsewhere in the review, completely unrelated
to the sentence I picked because it sounded relevant to negation.

> "capsule: the much anticipated re-adaptation..." — actual: negative,
> predicted: positive (score 6.0)

Six matches across the whole review — "good" four times, "interesting"
once, "nice" once — none of them near each other, none of them about
the film as a whole. A word-counting scorer with a 84-word dictionary
mostly isn't measuring sentiment, it's measuring how many times a
handful of common words happen to appear, which correlates with
sentiment about as well as you'd expect.

The pattern in all three: with 1.2% coverage, most "failures" aren't
the negation logic getting outsmarted by a clever sentence. They're a
handful of scattered, mostly unrelated word hits standing in for the
whole review, and the fix isn't better negation handling — it's a
lexicon more than 30 words long.

## Design notes

Reviews for the same book are stored under a generated key
(`Admin/<book>/<push key>`) rather than the book title, and `two.py`
averages sentiment and rating across all of a book's reviews at read
time — the schema is built around a book having many reviews from
different readers.

Negation, intensifiers, and diminishers stay active for a small window
of tokens after they appear, rather than only affecting the single
token right after them, so phrases like "not very good" land on the
negative side.

There's no POS tagging. The dictionaries are plain word and phrase
strings, not tagged by part of speech, so tagging every token as noun,
verb, adjective, etc. and then never checking that tag would just be
paying for work the scorer doesn't use. Actually using POS (scoring
only adjectives/adverbs, or splitting entries like "like" the verb from
"like" the preposition) would mean redesigning the dictionary format
around it — noted below as future work.

## What I'd do differently

- The lexicon needs to be roughly 50-100x bigger to cover normal
  writing. Hand-building word lists doesn't scale; I'd pull from an
  existing sentiment lexicon (like the one VADER ships with) instead of
  typing out 84 words by hand.
- Negation as a fixed token window is a hack, and the "isn't nearly as
  dull as this" example above shows where it falls over. Real
  dependency parsing would catch scope properly; that's a much bigger
  dependency for a small project though.
- Word-counting can't see "good parts, bad movie" — that needs
  something that understands the review has structure, not just a bag
  of tagged tokens.
- If I were doing this now I'd start with TF-IDF + logistic regression
  as the baseline, not the finish line — the table above says that
  plainly.

## Running it

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt movie_reviews vader_lexicon

cp .env.example .env   # fill in your Firebase project's config

python extract.py                          # prompts for genre/book/comment/rating
python extract.py --genre sci-fi --book "Dune" --comment "loved it" --rating 5

python two.py --genre sci-fi                # ranks books in a genre

pytest tests/ -v                             # unit tests
python evaluate.py                           # the results table above
```

Requires a Firebase Realtime Database if you want `extract.py` and
`two.py` to actually persist anything. `sentiment.py` and `evaluate.py`
don't need Firebase at all — `score_text()` is a plain function.
