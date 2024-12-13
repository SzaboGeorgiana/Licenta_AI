

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Încarcă datele de antrenare
dataset = load_dataset('json', data_files='training_data2.json')

# Împarte dataset-ul în seturi de antrenare și validare
dataset = dataset['train'].train_test_split(test_size=0.1)

# Încarcă modelul și tokenizer-ul
model_name = "Salesforce/codet5-base"

# Folosește AutoTokenizer și AutoModel pentru generalitate
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)  # Înlocuiește T5Tokenizer cu AutoTokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Înlocuiește T5ForConditionalGeneration cu AutoModel

# Pre-procesare: tokenizare
def preprocess_function(examples):
    # Tokenizare cu padding și truncation
    model_inputs = tokenizer(
        examples["input"],
        max_length=512,  # Lungimea maximă acceptată de model
        padding="max_length",
        truncation=True
    )
    with tokenizer.as_target_tokenizer():  # Tokenizează output-urile corect
        labels = tokenizer(
            examples["output"],
            max_length=512,
            padding="max_length",
            truncation=True
        )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Setări pentru antrenare
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,  # Folosește FP16 pentru accelerare pe GPU
    logging_dir='./logs',  # Log-uri TensorBoard
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
)

# Pornește antrenarea
trainer.train()

# Salvează modelul antrenat
model.save_pretrained("./fine_tuned_codet5")
tokenizer.save_pretrained("./fine_tuned_codet5")
