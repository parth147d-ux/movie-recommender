import difflib

def recommend(movie, df, similarity):
    movie = movie.lower()

    # get closest match (handles typos + partial names)
    titles = df['title'].str.lower().tolist()
    match = difflib.get_close_matches(movie, titles, n=1, cutoff=0.5)

    if not match:
        return ["Movie not found"]

    matched_title = match[0]
    index = df[df['title'].str.lower() == matched_title].index[0]

    distances = similarity[index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in movie_list:
        recommendations.append(df.iloc[i[0]].title)

    return recommendations