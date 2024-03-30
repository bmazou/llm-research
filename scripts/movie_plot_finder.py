import pandas as pd


class MoviePlotFinder:
    def __init__(self, csv_file_path="data/movie-plots-transformed.csv"):
        self.plots = self.load_plots(csv_file_path)
        
    def load_plots(self, csv_file_path):
        plots = pd.read_csv(csv_file_path)
        plots.set_index('item_id', inplace=True)
        return plots

    def find_plot(self, item_id, restructure_output=True):
        if item_id not in self.plots.index:
            print(f"Error: item_id {item_id} not found")
            return None
        
        plot = self.plots.loc[item_id, 'Plot']
        title = self.plots.loc[item_id, 'Title']
        
        return f'Plot of "{title}": {plot}' if restructure_output else plot