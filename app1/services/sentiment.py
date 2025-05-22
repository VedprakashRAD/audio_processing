from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
from collections import Counter
from app1.utils.logging_config import logging

# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize SpaCy with TextBlob
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")

def vader_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.1:
        return "positive"
    elif compound < -0.1:
        return "negative"
    else:
        return "neutral"

def textblob_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def spacy_sentiment(text):
    doc = nlp(text)
    polarity = doc._.blob.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def ensemble_sentiment(text):
    try:
        sentiments = [
            vader_sentiment(text),
            textblob_sentiment(text),
            spacy_sentiment(text)
        ]
        vote_counts = Counter(sentiments)
        return vote_counts.most_common(1)[0][0]
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}")
        raise


