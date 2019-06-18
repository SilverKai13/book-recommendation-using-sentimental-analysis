"""Lexicon-based sentiment scorer for book reviews.

Splits a review into sentences/tokens, tags tokens against five YAML
dictionaries (positive, negative, intensifier, diminisher, negation), and
sums a score across the whole review.

Design decision: POS tagging removed.
`nltk.pos_tag` used to run on every sentence, but `DictionaryTagger` only
ever matched on raw word/phrase form and `value_of()` returned 0 for every
POS tag it saw -- the tags were computed and then thrown away. Actually
using POS (e.g. only scoring adjectives/adverbs, or splitting entries like
"like" the verb vs "like" the preposition) would mean reworking the YAML
dictionaries to be POS-aware, which is a bigger project than a cleanup pass.
So: pay no cost for something we don't use. If this project grows a real
need to disambiguate words by part of speech, that's the next thing to add,
with the dictionaries redesigned around it.
"""

import os

import nltk
import yaml

DICT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dicts')

# How many tokens a negation/intensifier/diminisher stays "in effect" for.
# The original code only ever looked one token back, so "not very good"
# scored as positive: by the time "good" was scored, the immediately
# preceding token was "very" (inc), and "not" had already been forgotten.
NEGATION_WINDOW = 3


class Splitter(object):

    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()

    def split(self, text):
        """
        input format: a paragraph of text
        output format: a list of lists of words.
            e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
        """
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences


class DictionaryTagger(object):

    def __init__(self, dictionary_paths):
        dictionaries = []
        for path in dictionary_paths:
            with open(path, 'r') as dict_file:
                dictionaries.append(yaml.safe_load(dict_file))

        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))

    def tag(self, sentences):
        """sentences: list of lists of words, as produced by Splitter.split"""
        tagged = []
        for sentence in sentences:
            # (word, lemma, tags) triples -- lemma == word since we don't lemmatize.
            tagged.append([(word, word, []) for word in sentence])
        return [self.tag_sentence(sentence) for sentence in tagged]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while i < N:
            j = min(i + self.max_key_size, N)  # avoid overflow
            tagged = False
            while j > i:
                expression_form = ' '.join([word[0] for word in sentence[i:j]]).lower()
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]]).lower()
                literal = expression_lemma if tag_with_lemmas else expression_form
                if literal in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token:  # conserve any previous taggings on a single token
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence


def value_of(sentiment):
    if sentiment == 'positive':
        return 1
    if sentiment == 'negative':
        return -1
    return 0


def sentence_score(sentence_tokens):
    """Sum the sentiment score of one tagged sentence.

    Iterative (not recursive -- the original blew the stack on long
    reviews, one frame per token). Negation/intensifier/diminisher tags
    stay active for NEGATION_WINDOW tokens instead of just the one
    immediately following them, so "not very good" is scored as negative
    instead of the "not" being silently dropped.
    """
    score = 0.0
    negation_ttl = 0
    inc_ttl = 0
    dec_ttl = 0
    for token in sentence_tokens:
        tags = token[2]
        is_modifier = False
        if 'inv' in tags:
            negation_ttl = NEGATION_WINDOW
            is_modifier = True
        if 'inc' in tags:
            inc_ttl = NEGATION_WINDOW
            is_modifier = True
        if 'dec' in tags:
            dec_ttl = NEGATION_WINDOW
            is_modifier = True

        if not is_modifier:
            token_score = sum(value_of(tag) for tag in tags)
            if inc_ttl > 0:
                token_score *= 2.0
            elif dec_ttl > 0:
                token_score /= 2.0
            if negation_ttl > 0:
                token_score *= -1.0
            score += token_score

        if negation_ttl > 0:
            negation_ttl -= 1
        if inc_ttl > 0:
            inc_ttl -= 1
        if dec_ttl > 0:
            dec_ttl -= 1
    return score


def sentiment_score(tagged_review):
    """tagged_review: list of tagged sentences, as produced by DictionaryTagger.tag"""
    return sum(sentence_score(sentence) for sentence in tagged_review)


_default_tagger = None


def _get_default_tagger():
    global _default_tagger
    if _default_tagger is None:
        _default_tagger = DictionaryTagger([
            os.path.join(DICT_DIR, 'positive.yml'),
            os.path.join(DICT_DIR, 'negative.yml'),
            os.path.join(DICT_DIR, 'inc.yml'),
            os.path.join(DICT_DIR, 'dec.yml'),
            os.path.join(DICT_DIR, 'inv.yml'),
        ])
    return _default_tagger


def score_text(text, splitter=None, dicttagger=None):
    """Score a raw review string. This is the entry point extract.py and
    evaluate.py both use."""
    if not text or not text.strip():
        return 0.0
    splitter = splitter or Splitter()
    dicttagger = dicttagger or _get_default_tagger()
    sentences = splitter.split(text)
    tagged_sentences = dicttagger.tag(sentences)
    return sentiment_score(tagged_sentences)
