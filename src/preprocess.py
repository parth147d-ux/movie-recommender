import pandas as pd
import ast

def load_data():
    movies = pd.read_csv('data/tmdb_5000_movies.csv')
    credits = pd.read_csv('data/tmdb_5000_credits.csv')
    
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
    
    return movies


def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L


def convert_cast(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


def preprocess():
    movies = load_data()
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert_cast)
    movies['crew'] = movies['crew'].apply(fetch_director)

    movies['overview'] = movies['overview'].apply(lambda x: x.split())

   
    movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
    movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

    
    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

    new_df = movies[['movie_id','title','tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

    return new_df