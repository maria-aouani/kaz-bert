from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

app = Flask(__name__)

MODEL_PATH = "./kazakh_bert_finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForMaskedLM.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def correct_sentence(sentence, top_k=1):
    # Add [MASK] at the end of each word
    words = sentence.split()
    masked_sentence = " ".join([word + "[MASK]" for word in words])
    # Tokenize and predict
    inputs = tokenizer(masked_sentence, return_tensors="pt").to(device)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

    with torch.no_grad():
        outputs = model(**inputs)

    # Replace each [MASK] with the top prediction
    corrected_words = []
    for i, word in enumerate(words):
        mask_logits = outputs.logits[0, mask_token_index[i], :]
        predicted_token_id = torch.topk(mask_logits, top_k).indices[0].item()
        predicted_ending = tokenizer.decode([predicted_token_id]).strip()
        predicted_ending = predicted_ending.replace('##', '')
        corrected_words.append(word + predicted_ending)

    return " ".join(corrected_words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/correct', methods=['POST'])
def correct():
    input_text = request.form['input_text']
    corrected_text = correct_sentence(input_text)
    return jsonify({"corrected_text": corrected_text})

if __name__ == '__main__':
    app.run(debug=True)
