import csv

import Levenshtein


class MoviePlotFinder:
    def __init__(self, csv_file_path="data/movie-plots.csv"):
        self.csv_file_path = csv_file_path
        self.movie_data = {}
        self.load_movie_data()

    def load_movie_data(self):
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                title = row['Title']
                self.movie_data[title] = row['Plot']

    def find_plot(self, movie_title, restructure_output=True, min_distance_threshold=3):
        min_distance = float('inf')
        nearest_title = None
        for title in self.movie_data.keys():
            dist = Levenshtein.distance(movie_title.lower(), title.lower())

            found_perfect_match = dist == 0
            if found_perfect_match: 
                nearest_title = title
                break
            
            if dist < min_distance:
                min_distance = dist
                nearest_title = title
        
        if min_distance > min_distance_threshold:
            return None
        
        plot = self.movie_data[nearest_title]
        return f'Plot of "{nearest_title}": {plot}' if restructure_output else plot