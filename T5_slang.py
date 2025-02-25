from happytransformer import HappyTextToText, TTSettings, TTTrainArgs
import csv
from datasets import load_dataset

# Initialize the model with mT5 (this will load it to the CPU by default)
happy_tt = HappyTextToText("T5", "t5-base")

# Training arguments
train_args = TTTrainArgs(
    batch_size=1,
    max_input_length=512,
    max_output_length=512,
    num_train_epochs=3,     # Number of epochs
    learning_rate=5e-5
)

# Train the model (training code)
happy_tt.train("slang_sentences.csv", args=train_args)

# Save the fine-tuned model
happy_tt.save("fine_tuned_t5_model")

'''
# Evaluate the model after training
eval_result = happy_tt.eval("eval.csv")
print("Evaluation loss:", eval_result.loss)
'''

# Load the fine-tuned model (for future use)
# happy_tt = HappyTextToText("T5", "fine_tuned_mt5_model")

# Beam search settings for text generation
beam_settings = TTSettings(num_beams=5, min_length=1, max_length=50)

# Example sentences to correct
example_1 = "I don't know. Are you okay? Talk to you later. That was a great game. Goodbye!"
list_ex = example_1.split('.')
list_res = []
for sentence in list_ex:
    if sentence.strip():  # Skip empty sentences
        result_1 = happy_tt.generate_text(sentence, args=beam_settings)
        list_res.append(result_1.text)

corrected_text_1 = '. '.join(list_res) + '.'
print("Corrected Text 1:", corrected_text_1)

example_2 = "grammar: Меніңше, ауа райы ертең жақсы болады, бірақ жаңбыр жауады деп ойламаймын."
result_2 = happy_tt.generate_text(example_2, args=beam_settings)
print("Corrected Text 2:", result_2.text)
