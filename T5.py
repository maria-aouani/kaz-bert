from happytransformer import HappyTextToText, TTSettings, TTTrainArgs
import csv
import matplotlib.pyplot as plt
from datasets import load_dataset

happy_tt = HappyTextToText("T5", "t5-base")

train_dataset = load_dataset("jfleg", split='validation[:]')

eval_dataset = load_dataset("jfleg", split='test[:]')

for case in train_dataset["corrections"][:2]:
  print(case)
  print(case[0])
  print("--------------------------------------------------------")

def generate_csv(csv_path, dataset):
    with open(csv_path, 'w', newline='') as csvfile:
        writter = csv.writer(csvfile)
        writter.writerow(["input", "target"])
        for case in dataset:
            # Adding the task's prefix to input
            input_text = "grammar: " + case["sentence"]
            for correction in case["corrections"]:
                # a few of the cases contain blank strings.
                if input_text and correction:
                    writter.writerow([input_text, correction])


generate_csv("train.csv", train_dataset)
generate_csv("eval.csv", eval_dataset)
'''
before_result = happy_tt.eval("eval.csv")
print("Before loss:", before_result.loss)
'''
train_args = TTTrainArgs(
    batch_size=8,
    max_input_length=512,
    max_output_length=512,
    num_train_epochs=3,     # Added number of epochs
    learning_rate=2e-5,     # Typical learning rate
    logging_steps=10,       # Log every 10 steps
    eval_steps=50          # Evaluate every 50
)
happy_tt.train("train.csv", args=train_args)
'''
before_loss = happy_tt.eval("eval.csv")

print("After loss: ", before_loss.loss)
'''
beam_settings =  TTSettings(num_beams=5, min_length=1, max_length=50)
example_1 = "grammar: This sentences, has bads grammar and spelling!"
result_1 = happy_tt.generate_text(example_1, args=beam_settings)
print(result_1.text)

example_2 = "grammar: I am enjoys, writtings articles ons AI."

result_2 = happy_tt.generate_text(example_2, args=beam_settings)
print(result_2.text)

replacements = [
  (" .", "."),
  (" ,", ","),
  (" '", "'"),
  (" ?", "?"),
  (" !", "!"),
  (" :", "!"),
  (" ;", "!"),
  (" n't", "n't"),
  (" v", "n't"),
  ("2 0 0 6", "2006"),
  ("5 5", "55"),
  ("4 0 0", "400"),
  ("1 7-5 0", "1750"),
  ("2 0 %", "20%"),
  ("5 0", "50"),
  ("1 2", "12"),
  ("1 0", "10"),
  ('" ballast water', '"ballast water')
]
