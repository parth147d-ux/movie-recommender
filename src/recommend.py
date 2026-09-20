"""
Title lookup + recommendation logic.

Title matching is deliberately forgiving: case, punctuation, leading
articles and spelling mistakes should all still find the right movie.
The ladder of strategies is, in order:

  1. exact match on the normalised title      "THE DARK KNIGHT"  -> hit
  2. article-insensitive match                "dark knight, the" -> hit
  3. prefix match                             "dark knight ri"   -> hit
  4. substring match                          "godfather"        -> hit
  5. fuzzy match (difflib, descending cutoff) "dark knigth"      -> hit
  6. token-overlap fallback                   "batman joker dark"-> hit

Within a tier, ties are broken by popularity, so "Batman" returns the
best-known Batman film rather than an obscure one.
"""

import difflib
import re

import numpy as np

from .model import weighted_rating
from .preprocess import normalize_title

_ARTICLES = ("the ", "a ", "an ")


def _strip_article(t):
    for art in _ARTICLES:
        if t.startswith(art):
            return t[len(art):]
    # handle the "Godfather, The" style
    m = re.match(r"^(.*),\s*(the|a|an)$", t)
    if m:
        return m.group(1)
    return t


def _best_by_popularity(df, idxs):
    """Of several candidate rows, return the most popular one's index."""
    if not len(idxs):
        return None
    sub = df.loc[list(idxs)]
    return int(sub["popularity"].astype(float).idxmax())


def find_movie_index(query, df):
    """
    Resolve a user-typed string to a row index of `df`, or None.

    Returns (index, matched_title) on success, (None, None) on failure.
    """
    q = normalize_title(query)
    if not q:
        return None, None

    titles = df["title_norm"].tolist()

    # 1. exact
    hits = df.index[df["title_norm"] == q]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]

    # 2. ignoring leading/trailing articles
    q_bare = _strip_article(q)
    bare = df["title_norm"].apply(_strip_article)
    hits = df.index[bare == q_bare]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]

    # 3. prefix
    hits = df.index[df["title_norm"].str.startswith(q)]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]

    # 4. substring (only for queries long enough to be meaningful)
    if len(q) >= 4:
        hits = df.index[df["title_norm"].str.contains(re.escape(q), regex=True)]
        if len(hits):
            i = _best_by_popularity(df, hits)
            return i, df.at[i, "title"]

    # 5. fuzzy - tighten first, then loosen, so we prefer close matches
    for cutoff in (0.85, 0.75, 0.65, 0.55, 0.45):
        matches = difflib.get_close_matches(q, titles, n=5, cutoff=cutoff)
        if matches:
            hits = df.index[df["title_norm"].isin(matches)]
            i = _best_by_popularity(df, hits)
            return i, df.at[i, "title"]

    # 6. token overlap - catches partial/reordered multi-word queries
    q_tokens = set(q.split())
    if q_tokens:
        def overlap(t):
            toks = set(str(t).split())
            return len(q_tokens & toks) / len(q_tokens)

        scores = df["title_norm"].apply(overlap)
        if scores.max() >= 0.5:
            best = df.index[scores == scores.max()]
            i = _best_by_popularity(df, best)
            return i, df.at[i, "title"]

    return None, None


def suggest_titles(query, df, n=5):
    """Closest title strings to `query` — used for 'did you mean...'."""
    q = normalize_title(query)
    if not q:
        return []
    matches = difflib.get_close_matches(q, df["title_norm"].tolist(), n=n, cutoff=0.3)
    out = []
    for m in matches:
        rows = df.index[df["title_norm"] == m]
        i = _best_by_popularity(df, rows)
        if i is not None:
            out.append(df.at[i, "title"])
    return out


def recommend(
    movie,
    df,
    similarity,
    top_n=5,
    quality_weight=0.25,
    candidate_pool=60,
    quality=None,
):
    """
    Recommend movies similar to `movie`.

    Strategy: pull the `candidate_pool` most similar movies by content, then
    re-rank them by a blend of similarity and quality score. Pure similarity
    alone happily recommends forgotten direct-to-video sequels; the blend
    keeps relevance first but breaks ties toward films worth watching.

    quality_weight=0 gives pure content similarity.

    Returns (results, matched_title):
      results       - list of dicts: title, similarity, score, rating,
                      votes, year
      matched_title - the title the query actually resolved to, or None
    """
    idx, matched_title = find_movie_index(movie, df)
    if idx is None:
        return [], None

    if quality is None:
        quality = weighted_rating(df)

    sims = np.asarray(similarity[idx], dtype=np.float32).copy()
    sims[idx] = -1.0  # never recommend the movie itself

    pool = min(candidate_pool, len(sims) - 1)
    cand = np.argpartition(-sims, pool)[:pool]
    cand = cand[sims[cand] > 0]
    if cand.size == 0:
        return [], matched_title

    cand_sims = sims[cand]
    lo, hi = float(cand_sims.min()), float(cand_sims.max())
    sim_norm = (
        (cand_sims - lo) / (hi - lo) if hi - lo > 1e-9 else np.ones_like(cand_sims)
    )

    final = (1.0 - quality_weight) * sim_norm + quality_weight * quality[cand]

    order = np.argsort(-final)[:top_n]  # positions within `cand`

    results = []
    for pos in order:
        i = int(cand[pos])
        year = df.at[i, "release_year"]
        results.append(
            {
                "title": df.at[i, "title"],
                "similarity": round(float(sims[i]), 4),
                "score": round(float(final[pos]), 4),
                "rating": round(float(df.at[i, "vote_average"]), 1),
                "votes": int(df.at[i, "vote_count"]),
                "year": None if year != year else int(year),  # NaN check
            }
        )
    return results, matched_title
