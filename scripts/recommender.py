import pandas as pd

from scripts.ease import EASE
from scripts.utils import Utils


class Recommender:
    def __init__(self, ratings_path="data/ratings.csv", movies_path="data/movies.csv"):
        self.ease = EASE()
        self.ratings = pd.read_csv(ratings_path)
        self.movies = pd.read_csv(movies_path)

    @staticmethod
    def get_movie_id(movies, movie_name):
        best_distance = float('inf')
        best_match = None
        for i, row in movies.iterrows():
            potential_name = row['title']
            distance = Utils.edit_distance(movie_name, potential_name)
            if distance < best_distance:
                best_distance = distance
                best_match = row['movieId']

        found_movie_name = movies[movies['movieId'] == best_match]['title'].values[0]
        found_movie_id = movies[movies['movieId'] == best_match]['movieId'].values[0]
        print(f'Looking for "{movie_name}". Found "{found_movie_name}"')
        return found_movie_id

    @staticmethod
    def get_df(ratings, movies, user_ratings, user_id):

        df = ratings.copy()
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df.drop('timestamp', axis=1, inplace=True)

        for movie, rating in user_ratings:
            movie_id = Recommender.get_movie_id(movies, movie)
            df = pd.concat([df, pd.DataFrame({'user_id': [user_id], 'item_id': [movie_id], 'rating': [rating]})])

        return df[['user_id', 'item_id', 'rating']]


    def merge_with_movie_names(self, predictions, movies):
        movie_names = movies.copy().drop('genres', axis=1)
        movie_names.columns = ['item_id', 'title']
        return predictions.merge(movie_names, on='item_id').sort_values('score', ascending=False)
    
    def predict_most_popular(self, k=10):
        return self.ratings.groupby('movieId').size().sort_values(ascending=False).head(k).reset_index(name='score')

    def fit_and_predict(self, user_ratings, user_id=611, k=10):
        no_user_ratings_provided = len(user_ratings) == 0
        if no_user_ratings_provided:
            predictions = self.predict_most_popular(k)
            predictions.columns = ['item_id', 'score']
        else:
            df = self.get_df(self.ratings, self.movies, user_ratings, user_id)
            self.ease.fit(df, implicit=False)
            predictions = self.ease.predict(df, [user_id], df.item_id.unique(), k)
        
        return self.merge_with_movie_names(predictions, self.movies) 
    
    
    def fit(self, model_path=None):
        df = self.ratings.copy()
        df.columns = ['user_id', 'item_id', 'rating', 'timestamp']
        df.drop('timestamp', axis=1, inplace=True)
        self.ease.fit(df, implicit=False, model_path=model_path)
        
    def predict(self, user_ratings, k=10):
        user_ratings = [(self.get_movie_id(self.movies, movie), rating) for movie, rating in user_ratings]
        
        predictions =  self.ease.predict(user_ratings, k)
        return self.merge_with_movie_names(predictions, self.movies)