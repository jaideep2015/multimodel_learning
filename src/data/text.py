"""Fixed Federal Reserve speech text: loading and cleaning.

The original lab uses a single Fed speech (the most recent one as of March
2024) as macro-economic context, applied identically to every loan sample --
it is not a per-loan text feature. See the README "Limitations" section.
"""

import string

import nltk
import pandas as pd


def load_fed_speeches(csv_path: str) -> pd.DataFrame:
    fed_speech = pd.read_csv(csv_path, delimiter=",", on_bad_lines="skip", engine="python")
    fed_speech["date"] = pd.to_datetime(fed_speech["date"], errors="coerce")
    fed_speech["year"] = fed_speech["date"].dt.year
    fed_speech["month"] = fed_speech["date"].dt.month
    return fed_speech


def select_speech(fed_speech: pd.DataFrame, year: int, month: int) -> str:
    """Return the most recent speech text matching the given year/month."""
    matches = fed_speech[(fed_speech.year == year) & (fed_speech.month == month)]
    return matches["text"].values[-1]


def clean_text(text: str) -> str:
    """Lowercase, tokenize, and strip stopwords/punctuation."""
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    table = str.maketrans("", "", string.punctuation)

    text = text.lower()
    tokens = word_tokenize(text)
    filtered_tokens = [token.translate(table) for token in tokens if token.isalnum() and token not in stop_words]
    return " ".join(filtered_tokens)


def get_fixed_speech(csv_path: str, year: int = 2024, month: int = 3) -> str:
    """End-to-end: load speeches, pick the target month, clean the text."""
    fed_speech = load_fed_speeches(csv_path)
    raw_text = select_speech(fed_speech, year, month)
    return clean_text(raw_text)
