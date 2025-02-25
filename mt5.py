from happytransformer import HappyTextToText, TTSettings, TTTrainArgs
import csv
from datasets import load_dataset

# Initialize the model with mT5 (this will load it to the CPU by default)
happy_tt = HappyTextToText("MT5", "google/mt5-base")

# Explicitly move the model to CPU (this step is optional since it's the default device)
#happy_tt.model.to("cpu")  # Moves the model to CPU if needed

# Training arguments
train_args = TTTrainArgs(
    batch_size=8,
    max_input_length=512,
    max_output_length=512,
    num_train_epochs=3,     # Number of epochs
    learning_rate=2e-5,     # Learning rate
    logging_steps=10,       # Log every 10 steps
    eval_steps=50           # Evaluate every 50 steps
)

# Train the model (training code)
happy_tt.train("train_kazakh_2_corrected.csv", args=train_args)
'''
# Evaluate the model after training
eval_result = happy_tt.eval("eval.csv")
print("Evaluation loss:", eval_result.loss)
'''
# Beam search settings for text generation
beam_settings = TTSettings(num_beams=5, min_length=1, max_length=50)

# Example sentences to correct
example_1 = "grammar: Менің атым Айжан, мен Алматыда тұрамын."
result_1 = happy_tt.generate_text(example_1, args=beam_settings)
print("Corrected Text 1:", result_1)

example_2 = "grammar: Мұнда менің айтарым бар, бірақ мен қазір ғана бастадым."
result_2 = happy_tt.generate_text(example_2, args=beam_settings)
print("Corrected Text 2:", result_2)