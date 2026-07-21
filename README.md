# book-recommendation-using-sentimental-analysis

A small tool that reads a book review and decides whether it sounds positive
or negative, then uses that to rank books within a genre alongside their star
ratings. Reviews and ratings are stored in Firebase, Google's cloud database.

I wrote this early in my degree, mostly to see if I could build a sentiment
scorer from scratch instead of importing one. It works, in the sense that it
runs and produces a number. Whether that number means anything is a separate
question, and this README has an honest answer: mostly no. See the results
section.

## How the scoring works

No machine learning, no pretrained model. The approach is a lexicon: five
lists of words and short phrases, each tagged as `positive`, `negative`,
`inc` (an intensifier, like "very"), `dec` (a diminisher, like "barely"), or
`inv` (a negation, like "not").

A review gets broken into sentences and individual words. Each word is
checked against the lists. Positive words add 1 to the score, negative words
subtract 1. An intensifier doubles whatever comes right after it, a
diminisher halves it, and a negation flips the sign. Add it all up and you
get one number per review — positive score, positive review.

`sentiment.py` has the scorer as a reusable piece of code (`score_text`).
`extract.py` is the command-line tool that takes a review and saves it to
Firebase. `two.py` reads everything back and ranks books within a genre.

## Results

I tested the scorer against a well-known set of 2,000 movie reviews from
NLTK (a popular language-processing toolkit), split evenly into positive and
negative. Not book reviews, but it's a labeled dataset that's easy to get and
old enough to fit the project's era, and reviews are reviews for this
purpose. A review counts as "positive" if its score is above zero.

| method                                     | accuracy | precision |
|---------------------------------------------|---------:|----------:|
| **This project's lexicon**                   |    0.582 |     0.565 |
| VADER (a well-known, professionally built lexicon scorer) | 0.635 | 0.598 |
| TF-IDF + Logistic Regression (a standard machine-learning approach) | 0.825 | 0.812 |

A coin flip gets you 0.50 accuracy. My lexicon beats that by a little. VADER,
which uses the same basic approach but with a much bigger, carefully built
word list, beats mine by five points. A standard machine-learning method —
nothing exotic, just counting word frequency and fitting a simple classifier
— beats both by a wide margin. You can reproduce this table by running
`python evaluate.py`.

The more useful number, though, is **coverage: 1.2%**. Only 1.2% of the
words in a typical review match anything in my dictionaries. The other
98.8% of every review is invisible to this scorer — it simply doesn't
recognize those words at all. The scoring logic itself isn't broken; there
just isn't enough vocabulary behind it. My lists have 30 positive words and
54 negative words. VADER's has about 7,500. That gap explains most of the
difference in the table above.

### Where it fails

Pulled straight from the mistakes the scorer made, then traced through to
see which words actually caused them:

> "stalked does not provide much suspense, though that is what it sets
> out to do..." (a ~40-sentence review) — actually negative, scored as
> positive (1.0)

"suspense" isn't in either word list — I checked, and my first guess about
why this one failed was wrong. What actually happens: out of roughly 500
words in the review, only five ever match anything at all: "love",
"contrary", two separate instances of "not ... bad", and one plain "good",
scattered across unrelated sentences. One of the "not bad"s is scored
correctly as negative; the other is scored as positive, because too many
unrelated words sit between "not" and "bad" for the negation rule to still
apply by the time it reaches "bad". Five essentially random sign flips
aren't a meaningful signal — they're noise that happens to add up to a
number. This is a coverage problem, not a negation problem.

> "...isn't nearly as dull as this" — actually negative, scored as
> positive (2.0)

Same shape of failure. "dull" isn't in either word list, so the negation
logic never even gets triggered here — that phrase contributes nothing to
the score either way. The +2 actually comes from "interesting" and "too
good" appearing elsewhere in the review, completely unrelated to the
sentence that looked, at a glance, like the interesting case.

> "capsule: the much anticipated re-adaptation..." — actually negative,
> scored as positive (6.0)

Six matches across the whole review — "good" four times, "interesting"
once, "nice" once — none of them near each other, none of them actually
about the film as a whole. With only 84 words in its entire vocabulary,
this scorer mostly isn't measuring sentiment. It's measuring how many times
a handful of common words happen to show up, which correlates with actual
sentiment about as well as you'd expect from that.

The pattern across all three examples: with only 1.2% coverage, most of the
mistakes aren't the negation rule being outsmarted by a clever sentence.
They're a handful of scattered, mostly irrelevant word matches standing in
for the whole review. The fix isn't smarter negation handling — it's a
vocabulary more than 30 words long.

## Design notes

Reviews for the same book are stored under a generated key
(`Admin/<book>/<push key>`) rather than under the book's title, and `two.py`
averages sentiment and rating across all of a book's reviews when it reads
them back. The whole structure assumes a book will have many reviews from
different readers, not just one.

Negation, intensifiers, and diminishers stay "active" for a few words after
they appear, rather than only affecting the single word right after them.
That's what lets a phrase like "not very good" correctly land on the
negative side.

There's no part-of-speech tagging (labeling each word as a noun, verb,
adjective, and so on). The word lists are just plain text, not tagged by
part of speech, so labeling every word that way and then never using the
label would just be extra work for nothing. Actually using it — for
example, only scoring adjectives and adverbs, or telling apart different
meanings of the same word — would mean redesigning the word lists around
it. That's noted below as future work.

## What I'd do differently

- The vocabulary needs to be roughly 50 to 100 times bigger to cover normal
  writing. Building word lists by hand doesn't scale; I'd pull from an
  existing sentiment lexicon (like the one VADER uses) instead of typing
  out 84 words myself.
- Negation as a fixed word-count window is a shortcut, and the "isn't
  nearly as dull as this" example above shows where it breaks down. Proper
  grammatical parsing would handle this correctly, but that's a much
  heavier dependency for a small project.
- Simple word-counting can't tell "good parts, bad movie" from "bad parts,
  good movie" — that requires understanding how a review is structured, not
  just adding up tagged words.
- If I were starting this today, I'd begin with the TF-IDF and logistic
  regression approach as the baseline, not treat it as the advanced option
  — the results table above makes that case clearly enough on its own.

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

You'll need a Firebase Realtime Database if you want `extract.py` and
`two.py` to actually save and retrieve anything. `sentiment.py` and
`evaluate.py` don't need Firebase at all — `score_text()` works as a
standalone function.
