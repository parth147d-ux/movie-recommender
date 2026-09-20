# 🎬 Movie Recommender System

A content-based movie recommender built on the TMDB 5000 Movie Dataset. Each
movie is represented as a bag of "tags" drawn from its plot overview, genres,
keywords, top-billed cast and director; similarity between movies is measured
by cosine similarity over TF-IDF vectors, and candidates are re-ranked using an
IMDB-style weighted rating so that results are both relevant and worth
watching.

## Features

- **Forgiving search.** Case, punctuation, extra whitespace, leading articles,
  partial titles and spelling mistakes all still resolve. `dark knigth`,
  `THE DARK KNIGHT`, `spiderman 3` and `godfather` all find the right film.
  Unmatched queries return "did you mean" suggestions.
- **TF-IDF over bag-of-words**, with a toggle to switch back to plain counts so
  the two can be compared directly.
- **Quality-aware re-ranking.** The top ~60 most similar films are re-scored
  with a blend of similarity and weighted rating, so an obscure 9.4-from-11-
  votes entry doesn't outrank a genuinely comparable film. The weight is
  adjustable, and 0 gives pure content similarity.
- **Weighted tag fields.** Genre, cast and director tokens are repeated so
  structural signals count for more than a single incidental plot word.
- **Cached pipeline.** The dataset parse and similarity matrix are computed
  once, not on every button press.

## Setup

```bash
git clone https://github.com/parth147d-ux/movie-recommender
cd movie-recommender
pip install -r requirements.txt
```

Download the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
and place both CSVs in `data/`:

```
data/tmdb_5000_movies.csv
data/tmdb_5000_credits.csv
```

## Usage

```bash
# Web UI
python -m streamlit run app/app.py

# Command line
python main.py
python main.py "dark knigth"

# Smoke tests (generates its own synthetic data, no dataset needed)
python tests/test_pipeline.py
```

## Project structure

```
movie-recommender/
├── app/app.py              Streamlit interface
├── src/preprocess.py       Loading, parsing, feature engineering
├── src/model.py            TF-IDF vectorizer, cosine similarity, weighted rating
├── src/recommend.py        Title resolution + recommendation ranking
├── main.py                 CLI entry point
└── tests/test_pipeline.py  End-to-end smoke tests
```

## How it works

1. **Load and merge** `movies.csv` and `credits.csv` on title.
2. **Parse** the stringified JSON columns (`genres`, `keywords`, `cast`, `crew`)
   into token lists, taking the top 3 cast members and the director.
3. **Normalise** tokens — lowercase, strip internal spaces so `Science Fiction`
   becomes one token, then apply stemming so `fights`/`fighting` collapse.
4. **Vectorise** the combined tag string with TF-IDF and compute the full
   pairwise cosine similarity matrix.
5. **Recommend** by resolving the query to a row, taking the most similar
   candidates, and re-ranking by `(1-w) · similarity + w · quality`.

### A note on indexing

`dropna()` removes rows but leaves the original index labels behind, so the
index becomes gapped (`0, 1, 3, 7, ...`). The similarity matrix, being a NumPy
array, is always indexed positionally `0..N-1`. Looking a movie up by its
index *label* and using that number to index the matrix therefore silently
returns another movie's similarity row. `preprocess()` calls
`reset_index(drop=True)` after dropping so the two stay aligned — worth knowing
if you adapt this code, because the failure mode is wrong answers rather than
an error.

<<<<<<< HEAD
## 🖥️ UI Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/581c21f7-5adc-4ca1-ba02-b38c54e85710" width="800"/>
</p>


=======
## Limitations

These are inherent to the approach rather than bugs:

- **No personalisation.** This is content-based, not collaborative filtering —
  the same input always produces the same output regardless of who is asking.
  There is no user model and no feedback loop.
- **Cold start.** A film with no overview, keywords or credits in the dataset
  cannot be recommended meaningfully.
- **Static dataset.** TMDB 5000 ends in 2017, so nothing more recent exists.
- **Bag-of-words has no semantics.** "Space" and "cosmos" are unrelated tokens
  under TF-IDF. Sentence embeddings over the overview text would capture this;
  that's the natural next iteration.
- **No formal evaluation.** Content-based recommenders are hard to score
  without interaction data. Quality is currently assessed by inspection rather
  than a held-out metric.

## Possible next steps
>>>>>>> e263770 (Updated movie recommender (new pipeline + UI + improvements))

- Swap TF-IDF for sentence embeddings (`sentence-transformers`) and compare.
- Add an EDA notebook covering genre distribution, rating spread and keyword
  frequency.
- Add poster art via the TMDB API.
- Deploy to Streamlit Community Cloud for a live demo link.
