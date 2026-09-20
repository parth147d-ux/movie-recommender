"""Smoke tests: build TMDB-shaped fake CSVs, run the whole pipeline."""
import json, os, sys, tempfile
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess, normalize_title
from src.model import create_model, weighted_rating
from src.recommend import recommend, find_movie_index, suggest_titles


def make_csvs(d):
    def g(names): return json.dumps([{"id": i, "name": n} for i, n in enumerate(names)])
    def crew(director): return json.dumps([{"job": "Director", "name": director},
                                           {"job": "Editor", "name": "Someone Else"}])
    rows = [
        # title, overview, genres, keywords, cast, director, vote_avg, votes, pop
        ("The Dark Knight", "Batman fights the Joker in Gotham city crime chaos",
         ["Action","Crime","Drama"], ["batman","joker","dc comics"],
         ["Christian Bale","Heath Ledger","Aaron Eckhart"], "Christopher Nolan", 8.2, 12000, 120.0),
        ("The Dark Knight Rises", "Batman returns to Gotham to fight Bane crime",
         ["Action","Crime","Drama"], ["batman","bane","dc comics"],
         ["Christian Bale","Tom Hardy","Anne Hathaway"], "Christopher Nolan", 7.6, 9000, 95.0),
        ("Batman Begins", "Bruce Wayne becomes Batman in Gotham fighting crime",
         ["Action","Crime","Drama"], ["batman","origin","dc comics"],
         ["Christian Bale","Michael Caine","Liam Neeson"], "Christopher Nolan", 7.5, 7000, 80.0),
        ("Batman & Robin", "Batman and Robin fight Mr Freeze in Gotham",
         ["Action","Crime"], ["batman","robin","dc comics"],
         ["George Clooney","Chris O'Donnell","Arnold Schwarzenegger"], "Joel Schumacher", 4.2, 900, 20.0),
        ("Inception", "A thief enters dreams to steal secrets in a heist",
         ["Action","Science Fiction","Thriller"], ["dream","heist","subconscious"],
         ["Leonardo DiCaprio","Joseph Gordon-Levitt","Ellen Page"], "Christopher Nolan", 8.1, 14000, 150.0),
        ("Interstellar", "Astronauts travel through a wormhole to save humanity",
         ["Adventure","Drama","Science Fiction"], ["space","wormhole","time"],
         ["Matthew McConaughey","Anne Hathaway","Jessica Chastain"], "Christopher Nolan", 8.0, 11000, 140.0),
        ("The Godfather", "The aging patriarch of a crime family transfers control",
         ["Drama","Crime"], ["mafia","family","crime"],
         ["Marlon Brando","Al Pacino","James Caan"], "Francis Ford Coppola", 8.5, 6000, 70.0),
        ("Obscure Crime Gem", "A gritty crime drama in a decaying city underworld",
         ["Crime","Drama"], ["crime","underworld"],
         ["Unknown Actor","Another Unknown","Third Unknown"], "Nobody Famous", 9.4, 11, 1.0),
        ("Spider-Man 3", "Peter Parker faces Venom and Sandman as Spider-Man",
         ["Action","Adventure"], ["superhero","marvel","venom"],
         ["Tobey Maguire","Kirsten Dunst","Topher Grace"], "Sam Raimi", 5.9, 4000, 60.0),
        ("Broken Row Movie", None,  # <- will be dropped by dropna, creating an index gap
         ["Drama"], ["nothing"], ["Nobody"], "Nobody", 5.0, 10, 2.0),
        ("Avatar", "A marine on an alien planet Pandora joins the Navi people",
         ["Action","Adventure","Science Fiction"], ["alien","planet","3d"],
         ["Sam Worthington","Zoe Saldana","Sigourney Weaver"], "James Cameron", 7.2, 11800, 185.0),
    ]
    movies, credits = [], []
    for i, (t, ov, gen, kw, cast, dirn, va, vc, pop) in enumerate(rows, start=1):
        movies.append({"id": i, "title": t, "overview": ov, "genres": g(gen),
                       "keywords": g(kw), "vote_average": va, "vote_count": vc,
                       "popularity": pop, "release_date": f"20{10+i%9:02d}-01-01"})
        credits.append({"movie_id": i, "title": t, "cast": g(cast), "crew": crew(dirn)})
    pd.DataFrame(movies).to_csv(os.path.join(d, "tmdb_5000_movies.csv"), index=False)
    pd.DataFrame(credits).to_csv(os.path.join(d, "tmdb_5000_credits.csv"), index=False)


def main():
    d = tempfile.mkdtemp()
    make_csvs(d)
    df = preprocess(d)

    print("=== preprocess ===")
    print("rows:", len(df), "| index is clean RangeIndex:",
          list(df.index) == list(range(len(df))))
    assert list(df.index) == list(range(len(df))), "INDEX GAP BUG STILL PRESENT"
    assert "Broken Row Movie" not in df["title"].values, "null-overview row not dropped"
    print("columns:", list(df.columns))
    print("sample tags:", df.at[0, "tags"][:110], "...")

    sim = create_model(df)
    q = weighted_rating(df)
    print("\nsimilarity shape:", sim.shape, "| diag ~1:", round(float(sim[0][0]), 3))
    assert sim.shape == (len(df), len(df))

    print("\n=== title matching (case / punctuation / spelling / partial) ===")
    queries = ["the dark knight", "THE DARK KNIGHT", "  the   DARK knight  ",
               "dark knigth", "Dark Knight", "spiderman 3", "SPIDER-MAN 3",
               "godfather", "the godfather", "intersteller", "incepton",
               "batman", "avatr", "zzzzzzqqqq nonsense"]
    for qq in queries:
        i, title = find_movie_index(qq, df)
        print(f"  {qq!r:28} -> {title}")

    print("\n=== recommendations ===")
    for qq in ["dark knigth", "inception", "godfather"]:
        res, matched = recommend(qq, df, sim, top_n=4, quality=q)
        print(f"\n  query {qq!r} matched -> {matched}")
        for r in res:
            print(f"    {r['title']:<24} sim={r['similarity']:.3f} "
                  f"score={r['score']:.3f} rating={r['rating']} votes={r['votes']}")

    print("\n=== quality weighting: does the 9.4/11-votes movie get suppressed? ===")
    pure, _ = recommend("the godfather", df, sim, top_n=4, quality_weight=0.0, quality=q)
    blend, _ = recommend("the godfather", df, sim, top_n=4, quality_weight=0.35, quality=q)
    print("  pure similarity :", [r["title"] for r in pure])
    print("  with quality    :", [r["title"] for r in blend])

    print("\n=== not-found path ===")
    res, matched = recommend("zzzzzzqqqqq", df, sim, quality=q)
    print("  results:", res, "| matched:", matched)
    print("  suggestions for 'dark':", suggest_titles("dark", df))

    print("\n=== self-exclusion check ===")
    for qq in ["inception", "avatar", "batman begins"]:
        res, matched = recommend(qq, df, sim, top_n=5, quality=q)
        assert matched not in [r["title"] for r in res], f"{matched} recommended itself!"
    print("  no movie recommends itself: OK")

    print("\n=== count vs tfidf comparison ===")
    sim_count = create_model(df, method="count")
    a, _ = recommend("the dark knight", df, sim_count, top_n=3, quality=q)
    b, _ = recommend("the dark knight", df, sim, top_n=3, quality=q)
    print("  count:", [r["title"] for r in a])
    print("  tfidf:", [r["title"] for r in b])

    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
