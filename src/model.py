"""
Similarity model + quality score for the movie recommender.

Two pieces:
  1. create_model()      -> content similarity matrix (TF-IDF + cosine)
  2. weighted_rating()   -> IMDB-style quality score per movie

recommend.py takes the top candidates by similarity and re-ranks them
using the quality score, so results are both relevant and watchable.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
<<<<<<< HEAD
@st.cache_resource
def create_model(df):
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(df['tags']).toarray()
=======

>>>>>>> e263770 (Updated movie recommender (new pipeline + UI + improvements))

def create_model(df, max_features=8000, method="tfidf", ngram_range=(1, 1)):
    """
    Build the movie-to-movie cosine similarity matrix from `df['tags']`.

<<<<<<< HEAD
    return similarity
=======
    method:
      "tfidf" (default) - down-weights words common across the whole corpus,
                          so generic plot words matter less than distinctive
                          ones. Generally gives better recommendations.
      "count"           - the plain bag-of-words baseline, kept so the two
                          can be compared side by side.

    Returns an (n_movies, n_movies) float32 numpy array. Row i corresponds
    to df.iloc[i] — positional, which is why preprocess() resets the index.
    """
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

    # cosine_similarity handles sparse input directly - no .toarray(), which
    # on 5000 x 8000 would waste a few hundred MB for no benefit.
    similarity = cosine_similarity(vectors).astype(np.float32)
    return similarity


def weighted_rating(df, quantile=0.70):
    """
    IMDB-style weighted rating, returned normalised to 0..1.

        WR = (v / (v + m)) * R + (m / (v + m)) * C

      R = the movie's own average rating
      v = its number of votes
      m = vote-count threshold (the given quantile of all vote counts)
      C = mean rating across all movies

    A movie with 9.0 from 12 votes gets pulled toward the global mean;
    a movie with 8.2 from 14,000 votes keeps its score. This stops
    obscure high-rated entries from dominating the re-ranking.
    """
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
>>>>>>> e263770 (Updated movie recommender (new pipeline + UI + improvements))
