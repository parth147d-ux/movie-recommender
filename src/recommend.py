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
    if not len(idxs):
        return None
    sub = df.loc[list(idxs)]
    return int(sub["popularity"].astype(float).idxmax())


def find_movie_index(query, df):
    q = normalize_title(query)
    if not q:
        return None, None

    titles = df["title_norm"].tolist()
    hits = df.index[df["title_norm"] == q]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]
    q_bare = _strip_article(q)
    bare = df["title_norm"].apply(_strip_article)
    hits = df.index[bare == q_bare]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]

    hits = df.index[df["title_norm"].str.startswith(q)]
    if len(hits):
        i = _best_by_popularity(df, hits)
        return i, df.at[i, "title"]

    if len(q) >= 4:
        hits = df.index[df["title_norm"].str.contains(re.escape(q), regex=True)]
        if len(hits):
            i = _best_by_popularity(df, hits)
            return i, df.at[i, "title"]

    for cutoff in (0.85, 0.75, 0.65, 0.55, 0.45):
        matches = difflib.get_close_matches(q, titles, n=5, cutoff=cutoff)
        if matches:
            hits = df.index[df["title_norm"].isin(matches)]
            i = _best_by_popularity(df, hits)
            return i, df.at[i, "title"]

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
  
    idx, matched_title = find_movie_index(movie, df)
    if idx is None:
        return [], None

    if quality is None:
        quality = weighted_rating(df)

    sims = np.asarray(similarity[idx], dtype=np.float32).copy()
    sims[idx] = -1.0
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

    order = np.argsort(-final)[:top_n]
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
