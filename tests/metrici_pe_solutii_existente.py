import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer
from evaluate import load
from tqdm import tqdm  # Pentru a adăuga un progress bar

# Încarcă baza de date
with open("../baze_de_date/json/training_data2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Pregătește datele de testare
inputs = [example["input"] for example in data]
references = [example["output"] for example in data]

# Încarcă modelul și tokenizer-ul
model_path = "../fine_tuned_codet5_metrici_1"  # Adaptează calea dacă este diferită
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Funcție pentru generarea codului
def generate_code(instruction):
    input_ids = tokenizer.encode(instruction, return_tensors="pt")
    outputs = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generează predicții
predictions = []
for instruction in tqdm(inputs, desc="Generating predictions"):
    predictions.append(generate_code(instruction))

# Încarcă metricile BLEU și ROUGE
bleu = load("bleu")
rouge = load("rouge")

# Calculează scorurile BLEU și ROUGE
bleu_score = bleu.compute(predictions=predictions, references=[[ref] for ref in references])
rouge_score = rouge.compute(predictions=predictions, references=references)

# Afișează rezultatele
print(f"BLEU Score: {bleu_score['bleu']}")
print("ROUGE Scores:")
for key, value in rouge_score.items():
    print(f"  {key}: {value:.4f}")
