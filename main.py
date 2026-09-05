from src.preprocess import preprocess
from src.model import create_model
from src.recommend import recommend

df = preprocess()
similarity = create_model(df)

movie_name = input("Enter movie name: ")
results = recommend(movie_name, df, similarity)

print("\nRecommended Movies:")
for movie in results:
    print(movie)