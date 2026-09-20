import ast
import os
import re
import pandas as pd
try:
    from nltk.stem import PorterStemmer

    _ps = PorterStemmer()

    def _stem(word):
        return _ps.stem(word)

except ImportError:
    _SUFFIXES = ("ing", "edly", "ly", "ies", "es", "ed", "s")

    def _stem(word):
        for suf in _SUFFIXES:
            if len(word) > len(suf) + 2 and word.endswith(suf):
                return word[: -len(suf)]
        return word


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_data(data_dir=DATA_DIR):
    """Load and merge the TMDB 5000 movies + credits CSVs."""
    movies_path = os.path.join(data_dir, "tmdb_5000_movies.csv")
    credits_path = os.path.join(data_dir, "tmdb_5000_credits.csv")

    for path in (movies_path, credits_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing dataset file: {path}\n"
                "Download the TMDB 5000 Movie Dataset from Kaggle and place "
                "both CSVs in the data/ folder."
            )

    movies = pd.read_csv(movies_path)
    credits = pd.read_csv(credits_path)
    movies = movies.merge(credits, on="title", suffixes=("", "_cr"))

    if "movie_id" not in movies.columns:
        for alt in ("id", "movie_id_cr", "id_cr"):
            if alt in movies.columns:
                movies["movie_id"] = movies[alt]
                break
        else:
            movies["movie_id"] = range(len(movies))

    keep = [
        "movie_id", "title", "overview", "genres", "keywords", "cast", "crew",
        "vote_average", "vote_count", "popularity", "release_date",
    ]
    keep = [c for c in keep if c in movies.columns]
    return movies[keep]

def _safe_literal_eval(text):
    """Parse a stringified list of dicts, returning [] on bad input."""
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def convert(text):
    """All `name` values from a stringified list of dicts."""
    return [d["name"] for d in _safe_literal_eval(text) if "name" in d]


def convert_cast(text, limit=3):
    """Top-N billed cast members."""
    return [d["name"] for d in _safe_literal_eval(text) if "name" in d][:limit]


def fetch_director(text):
    """Director name(s) from the crew field."""
    return [
        d["name"]
        for d in _safe_literal_eval(text)
        if d.get("job") == "Director" and "name" in d
    ][:1]

@st.cache_data
def preprocess():
    movies = load_data()
    movies.dropna(inplace=True)
    movies.reset_index(drop=True,inplace=True)

def normalize_title(title):
        if not isinstance(title, str):
        return ""
    t = title.lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9\s]", " ", t)     
    return re.sub(r"\s+", " ", t).strip()


def _clean_tokens(items):
    return [str(i).replace(" ", "").lower() for i in items]


    return new_df

def _stem_text(text):
    return " ".join(_stem(w) for w in text.split())




def preprocess(data_dir=DATA_DIR):

    movies = load_data(data_dir)

    movies = movies.dropna(subset=["title", "overview"]).copy()
    movies = movies.drop_duplicates(subset=["title"], keep="first")
    movies = movies.reset_index(drop=True)

    movies["genres"] = movies["genres"].apply(convert).apply(_clean_tokens)
    movies["keywords"] = movies["keywords"].apply(convert).apply(_clean_tokens)
    movies["cast"] = movies["cast"].apply(convert_cast).apply(_clean_tokens)
    movies["crew"] = movies["crew"].apply(fetch_director).apply(_clean_tokens)

    overview_tokens = movies["overview"].astype(str).str.lower().str.split()

    movies["tags"] = (
        overview_tokens
        + movies["genres"] * 2
        + movies["keywords"]
        + movies["cast"] * 2
        + movies["crew"] * 2
    )
    movies["tags"] = movies["tags"].apply(lambda toks: " ".join(toks))
    movies["tags"] = movies["tags"].apply(_stem_text)

    movies["title_norm"] = movies["title"].apply(normalize_title)

    if "release_date" in movies.columns:
        movies["release_year"] = pd.to_datetime(
            movies["release_date"], errors="coerce"
        ).dt.year
    else:
        movies["release_year"] = pd.NA

    for col in ("vote_average", "vote_count", "popularity"):
        if col not in movies.columns:
            movies[col] = 0.0
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0.0)

    cols = [
        "movie_id", "title", "title_norm", "tags",
        "vote_average", "vote_count", "popularity", "release_year",
    ]
    return movies[cols].reset_index(drop=True)

