from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer

# Încarcă modelul antrenat
model_path = "./fine_tuned_codet5"
# tokenizer = T5Tokenizer.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

model = T5ForConditionalGeneration.from_pretrained(model_path)

# Prompt-ul pentru generare
input_text = "Open chrome browser\nLoad the page: https://ancabota09.wixsite.com/intern\nCheck if the search button exists\nClick the search button."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generare cod
outputs = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
