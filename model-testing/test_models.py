from transformers import T5ForConditionalGeneration, T5Tokenizer


def ask_question(question, tokenizer, model):
    input_ids = tokenizer(question, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, max_length=200)
    return tokenizer.decode(outputs[0])

def load_questions(input_path):
    questions = []
    with open (input_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(":")
            q_num = parts[0]
            question = ":".join(parts[1:]).strip()

            questions.append((q_num, question))

    return questions

            

def main(model_type):
    model_name = f"google/flan-t5-{model_type}"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")

    
    input_path = "input/questions.txt"
    questions = load_questions(input_path)
    results = []
    
    for q_num, question in questions:
        print("Asking:", q_num)
        answer = ask_question(question, tokenizer, model)
        res = f'{q_num}: {question}\n\nAnswer: {answer}\n'
        results.append(res)
        
    with open(f"results/flan-t5/{model_type}.txt", "w") as f:
        for res in results:
            res = res.replace("<pad>", "").replace("</s>", "")
            f.write(res)
            f.write("-"*50)
            f.write("\n")



if __name__ == '__main__':
    model_type = "xl"
    main(model_type)