class Utils:
    @staticmethod
    def split_llm_answer(text):
        movie_part, general_part = text.split('---')
        movie_part = movie_part.strip()
        general_part = general_part.strip()

        movie_part = movie_part.split(';;;')

        movie_ratings = [(movie.rsplit(': ', 1)[0], float(movie.rsplit(': ', 1)[1])) for movie in movie_part]
        return movie_ratings, general_part

    @staticmethod
    def edit_distance(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]