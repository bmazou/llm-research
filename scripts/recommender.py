import Levenshtein
import pandas as pd

from scripts.ease import EASE


class Recommender:
    def __init__(self, ratings_path, movie_names_path):
        self.ease = EASE()
        self.movie_names = self.read_movies_names_to_df(movie_names_path)
        self.ratings = self.read_ratings_to_df(ratings_path)

    def read_ratings_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        return df[['user_id', 'item_id', 'rating']]

    def read_movies_names_to_df(self, file_path):
        df = pd.read_json(file_path, lines=True)
        return df[['item_id', 'title', 'popularity', 'imdbId']]

    @staticmethod
    def get_movie_id(movies, movie_name):
        # TODO Add distance threshold?

        best_distance = float('inf')
        best_match = None

        remove_year_from_title = lambda title: title[:-7]   # Release year is always the last 7 characters


        for i, row in movies.iterrows():
            potential_name = remove_year_from_title(row['title'])
            distance = Levenshtein.distance(movie_name.lower(), potential_name.lower())

            if distance == 0:
                best_match = row['item_id']
                break

            if distance < best_distance:
                best_distance = distance
                best_match = row['item_id']


        found_movie_name = movies[movies['item_id'] == best_match]['title'].values[0]
        found_movie_id = movies[movies['item_id'] == best_match]['item_id'].values[0]
        print(f'Looking for "{movie_name}". Found "{found_movie_name}"')
        return found_movie_id

    def fit(self, model_path=None, lambda_=0.5):
        self.ease.fit(self.ratings, implicit=False, model_path=model_path, lambda_=lambda_)

    def merge_with_movie_names(self, predictions):
        return predictions.merge(self.movie_names, on='item_id').sort_values('score', ascending=False)

    def predict_most_popular(self, k=10):
        # self.movie_names is already sorted by popularity
        # returns k random movies from 10*k most popular movies
        return self.movie_names[:10*k].sample(k)[['item_id', 'title', 'popularity']].rename(columns={'popularity': 'score'})


    def predict(self, user_ratings, k=10, inputs_are_ids=False):
        no_ratings_provided = len(user_ratings) == 0
        if no_ratings_provided:
            return self.predict_most_popular(k)

        user_ratings = user_ratings if inputs_are_ids else [(self.get_movie_id(self.movie_names, movie), rating) for movie, rating in user_ratings]

        predictions =  self.ease.predict(user_ratings, k)
        return self.merge_with_movie_names(predictions)