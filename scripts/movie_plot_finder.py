import csv

from scripts.utils import Utils


class MoviePlotFinder:
    def __init__(self, csv_file_path):
        self.csv_file_path = csv_file_path
        self.movie_data = {}
        self.load_movie_data()

    def load_movie_data(self):
        with open(self.csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                title = row['Title']
                self.movie_data[title] = row['Plot']

    def find_plot(self, movie_title, restructure_output=True):
        # To do if min_distance > threshold  (3-6?), return None
        min_distance = float('inf')
        closest_title = None
        for title in self.movie_data.keys():
            dist = Utils.edit_distance(movie_title.lower(), title.lower())
            if dist < min_distance:
                min_distance = dist
                closest_title = title
        if closest_title:
            plot = self.movie_data[closest_title]
            if restructure_output: return f'Plot of "{movie_title}":\n{plot}'
            else: return plot
        else:
            return None