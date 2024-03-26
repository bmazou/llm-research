import pandas as pd

from scripts.ease import EASE
from scripts.utils import Utils


class Recommender:
    def __init__(self, ratings_path, movie_names_path):
        self.ease = EASE()
        self.ratings = self.read_ratings_to_df(ratings_path)
        self.movie_names = self.read_movies_names_to_df(movie_names_path)
        
    def read_ratings_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        return df[['user_id', 'item_id', 'rating']]
        
    def read_movies_names_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        return df[['item_id', 'title', 'imdbId']]


    @staticmethod
    def get_movie_id(movies, movie_name):
        best_distance = float('inf')
        best_match = None
        for i, row in movies.iterrows():
            potential_name = row['title'][:-7]
            distance = Utils.edit_distance(movie_name, potential_name)
            if distance < best_distance:
                best_distance = distance
                best_match = row['item_id']

        found_movie_name = movies[movies['item_id'] == best_match]['title'].values[0]
        found_movie_id = movies[movies['item_id'] == best_match]['item_id'].values[0]
        print(f'Looking for "{movie_name}". Found "{found_movie_name}"')
        return found_movie_id

    def fit(self, model_path=None):
        self.ease.fit(self.ratings, implicit=False, model_path=model_path)

    def merge_with_movie_names(self, predictions):
        return predictions.merge(self.movie_names, on='item_id').sort_values('score', ascending=False)

    def predict_most_popular(self, k=10):
        return self.ratings.groupby('item_id').size().sort_values(ascending=False).head(k).reset_index(name='score')


    def predict(self, user_ratings, k=10):
        user_ratings = [(self.get_movie_id(self.movie_names, movie), rating) for movie, rating in user_ratings]

        predictions =  self.ease.predict(user_ratings, k)
        return self.merge_with_movie_names(predictions)