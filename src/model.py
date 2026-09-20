import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
@st.cache_resource
def create_model(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()


def create_model(df, max_features=8000, method="tfidf", ngram_range=(1, 1)):
    if method == "tfidf":
        vec = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
            sublinear_tf=True,
        )
    elif method == "count":
        vec = CountVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=ngram_range,
        )
    else:
        raise ValueError("method must be 'tfidf' or 'count'")

    vectors = vec.fit_transform(df["tags"])
    similarity = cosine_similarity(vectors).astype(np.float32)
    return similarity


def weighted_rating(df, quantile=0.70):
  
    v = df["vote_count"].astype(float).to_numpy()
    R = df["vote_average"].astype(float).to_numpy()

    C = float(np.nanmean(R)) if len(R) else 0.0
    m = float(np.quantile(v, quantile)) if len(v) else 0.0

    denom = v + m
    wr = np.where(denom > 0, (v / denom) * R + (m / denom) * C, C)

    lo, hi = float(wr.min()), float(wr.max())
    if hi - lo < 1e-9:
        return np.zeros_like(wr, dtype=np.float32)
    return ((wr - lo) / (hi - lo)).astype(np.float32)

