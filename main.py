"""
Command-line version of the recommender (handy for quick testing).

    python main.py
    python main.py "dark knigth"
"""

import sys

from src.model import create_model, weighted_rating
from src.preprocess import preprocess
from src.recommend import recommend, suggest_titles


def main():
    print("Loading data and building the model...")
    df = preprocess()
    similarity = create_model(df)
    quality = weighted_rating(df)
    print(f"Ready - {len(df):,} movies loaded.\n")

    # A query passed as an argument runs once and exits; otherwise prompt
    # in a loop so the model is only built a single time.
    one_shot = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None

    while True:
        if one_shot:
            query = one_shot
            print(f"Enter movie name: {query}")
        else:
            try:
                query = input("Enter movie name (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
        if not query or query.lower() in {"q", "quit", "exit"}:
            break

        results, matched = recommend(query, df, similarity, quality=quality)

        if matched is None:
            print(f"  No match for '{query}'.")
            tips = suggest_titles(query, df)
            if tips:
                print("  Did you mean: " + ", ".join(tips))
        else:
            print(f"\n  Because you searched '{matched}':")
            for i, r in enumerate(results, 1):
                year = f" ({r['year']})" if r["year"] else ""
                print(f"   {i}. {r['title']}{year}"
                      f"  [rating {r['rating']}, {r['votes']:,} votes,"
                      f" sim {r['similarity']:.3f}]")
        print()

        if one_shot:
            break


if __name__ == "__main__":
    main()
