from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    RobertaTokenizer
from datasets import load_dataset
from evaluate import load

# Încarcă datele de antrenare
dataset = load_dataset('json', data_files='training_data2.json')

# Împarte dataset-ul în seturi de antrenare și validare
dataset = dataset['train'].train_test_split(test_size=0.1)

# Încarcă modelul și tokenizer-ul
model_name = "Salesforce/codet5-base"

# Înlocuiește T5Tokenizer cu RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
model = T5ForConditionalGeneration.from_pretrained(model_name)


# Funcția de preprocesare
def preprocess_function(examples):
    # Tokenizare cu padding și truncation
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,  # Lungimea maximă acceptată de model
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )["input_ids"]
    model_inputs["labels"] = labels
    return model_inputs


# Preprocesează datele
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Încarcă metricele BLEU și ROUGE
bleu = load("bleu")
rouge = load("rouge")


# Funcția de calculare a metricilor
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Formatează datele pentru BLEU și ROUGE
    references = [[label] for label in decoded_labels]
    predictions = decoded_preds

    # Calculează BLEU
    bleu_results = bleu.compute(predictions=predictions, references=references)

    # Calculează ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=decoded_labels)

    # Returnează metricele combinate
    return {
        "bleu": bleu_results["bleu"],
        "rouge1": rouge_results["rouge1"],
        "rouge2": rouge_results["rouge2"],
        "rougeL": rouge_results["rougeL"]
    }


# Setări pentru antrenare
training_args = Seq2SeqTrainingArguments(
    output_dir="../results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Folosește FP16 pentru accelerare pe GPU
    logging_dir="../logs",
    logging_strategy="epoch"
)

# Instanțiază trainer-ul
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics  # Adaugă funcția de metrici
)

# Pornește antrenarea
trainer.train()

# Salvează modelul antrenat
model.save_pretrained("./fine_tuned_codet5_metrici_1")
tokenizer.save_pretrained("./fine_tuned_codet5_metrici_1")

# Testare pe setul de validare
predictions = trainer.predict(tokenized_datasets["test"])
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

# Calculează metricele pe setul de testare
metrics = compute_metrics((predictions.predictions, predictions.label_ids))
print("Evaluation Metrics:", metrics)
