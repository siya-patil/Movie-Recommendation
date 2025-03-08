import pandas as pd

def load_ratings():
    r_cols = ['user_id', 'movie_id', 'rating']
    return pd.read_csv('data/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")

def load_movies():
    m_cols = ['movie_id', 'title']
    return pd.read_csv('data/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")

def get_merged_data():
    ratings = load_ratings()
    movies = load_movies()
    return pd.merge(movies, ratings)

if __name__ == "__main__":
    df = get_merged_data()
    print(df.head())  # Test if it loads properly
