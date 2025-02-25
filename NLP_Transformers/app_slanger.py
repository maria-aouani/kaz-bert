from flask import Flask, request, render_template, jsonify
from happytransformer import HappyTextToText, TTSettings

app = Flask(__name__)

# Load the fine-tuned mT5 model
MODEL_PATH = "./fine_tuned_t5_model"  # Path where your fine-tuned model is saved
happy_tt = HappyTextToText("T5", MODEL_PATH)

# Beam search settings for text generation
beam_settings = TTSettings(num_beams=5, min_length=1, max_length=50)

def correct_text(input_text):
    sentences = input_text.split('.')
    corrected_sentences = []
    for sentence in sentences:
        if sentence.strip():
            result = happy_tt.generate_text(sentence, args=beam_settings)
            corrected_sentences.append(result.text)
    return '. '.join(corrected_sentences) + '.'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct():
    input_text = request.form['input_text']
    corrected_text = correct_text(input_text)
    return jsonify({"corrected_text": corrected_text})

if __name__ == '__main__':
    app.run(debug=True)
