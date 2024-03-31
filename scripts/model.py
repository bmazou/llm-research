import os
import textwrap

from transformers import AutoModelForCausalLM, AutoTokenizer


class Model:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct-v0.2", use_gpu=False, local_model_path=None):
        if local_model_path is None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name) if not use_gpu else AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name) if not use_gpu else AutoTokenizer.from_pretrained(model_name, torch_dtype="auto")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True) if not use_gpu else AutoModelForCausalLM.from_pretrained(local_model_path, local_files_only=True, torch_dtype="auto").to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path, local_files_only=True) if not use_gpu else AutoTokenizer.from_pretrained(local_model_path, local_files_only=True, torch_dtype="auto")
        self.use_gpu = use_gpu

    def save_model(self, model_dir):
        if not os.path.exists(model_dir):
            print(f"{model_dir} directory doesn't exist. Can't save the model.")
            return

        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        print(f"Model saved in {model_dir}.")


    @staticmethod
    def get_extraction_prompt(user_input):
        return f"""
<s>[INST] You will be given a total of 11 prompts from different people, each of which consisting of two parts: it contains opinions about specific movies and it contains some general preferences. Find the opinions about movies and convert them into a rating score on a scale between 1 and 5, where the rating is based on the sentiment about the movie. Answer in the form of <movie_name>: <rating_score> and separate them with a ";;;". Secondly, print out the general preferences. The second part of the answer cannot contain a name of any movie and is for general preferences only. Separate the first and second part of the answer with "---". If there are no opinions about movies or general opinions given in the prompt, leave that part of the answer empty and write "<EMPTY>" instead of the answer for that part. Do not recommend any movies and do not generate anything that doesn't adhere to the provided structure for your asnwers. [/INST]
Understood!</s>
[INST] Prompt 1/11 "I'm a sucker for action-packed thrillers like 'Mad Max: Fury Road' and I also like 'Blade Runner 2049', but I can't stand "The Handmaiden" and I'm not really fond of 'Inception'. 'Interstellar' is decent. I'd like too see something with a unique setting, for example set underwater or in outer space" [/INST]
Mad Max: Fury Road: 5;;;Blade Runner 2049: 4;;;The Handmaiden: 1;;;Inception: 2;;;Interstellar: 3 --- I want a unique settings, such as underwater or in outer space.</s>
[INST] Prompt 2/11 "I'd like to watch a movie that takes place in a forest" [/INST]
<EMPTY> --- I want to watch a movie in a forest setting.</s>
[INST] Prompt 3/11 "I absolutely despise 'The Notebook', but I adore classic romance films like 'Pride and Prejudice' and I liked 'Pride and Prejudice and Zombies'. However, I found 'Gone with the Wind' to be overly long and melodramatic. 'Casablanca' remains a timeless favorite."  [/INST]
The Notebook: 1;;;Pride and Prejudice: 5;;;Pride and Prejudice and Zombies: 4;;;Gone with the Wind: 2;;;Casablanca: 5 --- <EMPTY>. </s>
[INST] Prompt 4/11 "Sci-fi epics are my jam, especially 'Star Wars' and 'The Matrix'. I've never been a fan of horror flicks, so 'The Exorcist' is a hard pass for me."  [/INST]
Star Wars: 5;;;The Matrix: 5;;;The Exorcist: 1 --- <EMPTY>.</s>
[INST] Prompt 5/11 "I prefer documentaries that delve into historical events or scientific discoveries. Fictional dramas often feel too contrived for my taste."  [/INST]
<EMPTY> --- I'm more inclined towards documentaries exploring real-world subjects rather than fictional dramas.</s>
[INST] Prompt 6/11 "I'm a sucker for thought-provoking dramas like 'The Shawshank Redemption' and 'The Godfather'. 'Titanic' didn't quite live up to the hype for me, though. 'Forrest Gump' is a heartwarming classic."  [/INST]
The Shawshank Redemption: 5;;;The Godfather: 5;;;Titanic: 3;;;Forrest Gump: 5 --- <EMPTY>.</s>
[INST] Prompt 7/11 "I love movies like 500 Days of Summer and 'Eternal Sunshine of the Spotless Mind'. The Proposal was entertaining but forgettable. I'm not a fan of tearjerkers like 'The Fault in Our Stars' and I'm somewhere in the middle when it comes to 'Pride and Prejudice and Zombies' and 'One Cut of the Dead'.  [/INST]
500 Days of Summer: 5;;;Eternal Sunshine of the Spotless Mind: 5;;;The Proposal: 4;;;The Fault in Our Stars: 2;;;Pride and Prejudice and Zombies: 3;;;One Cut of the Dead: 3 --- <EMPTY>.</s>
[INST] Prompt 8/11 "I want to see something that takes place in a cave or on a deserted island. My favourite movies are The Room and The Princess Bride. I also like Spirited Away, but I hate Avatar: Way of the Water. I also like Pixar and Ghibli animation." [/INST]
The Room: 5;;;The Princess Bride: 5;;;Spirited Away: 4;;;Avatar: The Way of Water: 1 --- I'm looking for a movie set in a cave or on a deserted island. I like Pixar and Ghibli animation.</s>
[INST] Prompt 9/11 "I love films such as 'The Social Network' and 'Steve Jobs'. 'Bohemian Rhapsody' was pretty good, but 'Marie Antoinette' felt shallow and overly stylized." [/INST]
The Social Network: 5;;;Steve Jobs: 5;;;Bohemian Rhapsody: 4;;;Marie Antoinette: 2 --- <EMPTY>.</s>
[INST] Prompt 10/11 "As a fan of the fantasy genre, I'm drawn to epic tales of adventure and heroism like 'The Lord of the Rings' trilogy and 'Pan's Labyrinth'. These films excel in their world-building and mythic storytelling. However, 'Eragon' failed to capture the magic of its source material, feeling rushed and underdeveloped. I appreciate movies that transport me to fantastical realms while also exploring universal themes of courage and redemption." [/INST]
The Lord of the Rings: 5;;;Pan's Labyrinth: 5;;;Eragon: 2 --- I'm drawn to epic tales of adventure and heroism and films with great world-building and mythic storytelling that transport me to fantastical realms and explore themes of courage and redemption.
[INST] Prompt 11/11 "{user_input}" [/INST]
"""

    @staticmethod
    def get_plot_fitness_prompt(user_preferences, movie_plot):
        return f"""
    <s>[INST] You are given a list of person's movie preference and a plot of a movie. Rate on a scale from 1 to 5 how well the preferences fit the plot. Asnwer only with the single number and an explanation. Seperate the number and the explanation with "---", so the format of the answer is "<ratings_number> --- <explanation>."
    User preferences: "{user_preferences}".
    {movie_plot}. [/INST]
    """

    @staticmethod
    def split_llm_answer(text):
        # If movie_part is None, recommender will return generic recommendations
        # If general_part is None, we will not compare the general preferences with plots of the movies, but only return the RS recommendations

        try:
            movie_part, general_part = text.split('---')
        except ValueError:
            return None, None


        movie_part = movie_part.strip()
        # if general_part contains <EMPTY> then return None
        general_part = general_part.strip() if not "<EMPTY>" in general_part else None
        if "<EMPTY>" in movie_part:
            return None, general_part

        try:
            movie_part = movie_part.split(';;;')
            movie_ratings = [(movie.rsplit(': ', 1)[0], float(movie.rsplit(': ', 1)[1])) for movie in movie_part]
        except ValueError:
            return None, general_part

        return movie_ratings, general_part

    @staticmethod
    def wrap_text(self, text, width=90):
        lines = text.split('\n')
        wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
        wrapped_text = '\n'.join(wrapped_lines)
        return wrapped_text

    @staticmethod
    def get_answer(text):
        parts = text.split("[/INST]")
        return parts[-1].replace("</s>", "").strip()

    @staticmethod
    def print_instruction_answer(text):
        parts = text.split("[/INST]")
        answer = parts[-1].replace("</s>", "").strip()
        instruction = parts[-2].split("Prompt 11/11")[-1].strip()
        print(f"{instruction}\n\n{answer}")
        print("-" * 100)

    def generate(self, input_text, reformat_input=False, max_length=2000, temp=0.1, verbose=False):
            prompt = f"""<s>[INST] {input_text} [/INST]""" if reformat_input else input_text
            inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False) if not self.use_gpu else self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
            outputs = self.model.generate(**inputs,
                                        max_length=max_length,
                                        temperature=temp,
                                        do_sample=True)
            text = self.tokenizer.batch_decode(outputs)[0]

            if verbose: self.print_instruction_answer(self.wrap_text(text))

            return self.get_answer(text)