#
#
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from datasets import load_dataset
#
# # Încarcă datele de antrenare
# dataset = load_dataset('json', data_files='training_data2.json')
#
# # Împarte dataset-ul în seturi de antrenare și validare
# dataset = dataset['train'].train_test_split(test_size=0.1)
#
# # Încarcă modelul și tokenizer-ul
# model_name = "Salesforce/codet5-base"
#
# # Folosește T5Tokenizer, care este compatibil cu CodeT5
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# # tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=False)
#
# model = T5ForConditionalGeneration.from_pretrained(model_name)
#
# # Pre-procesare: tokenizare
# def preprocess_function(examples):
#     # Tokenizare cu padding și truncation
#     model_inputs = tokenizer(
#         examples["input"],
#         max_length=512,  # Lungimea maximă acceptată de model
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )
#     labels = tokenizer(
#         examples["output"],
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )["input_ids"]
#     model_inputs["labels"] = labels
#     return model_inputs
#
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
#
# # Setări pentru antrenare
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=3,
#     predict_with_generate=True,
#     fp16=True,  # Folosește FP16 pentru accelerare pe GPU
# )
#
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
# )
#
# # Pornește antrenarea
# trainer.train()
#
# # Salvează modelul antrenat
# model.save_pretrained("./fine_tuned_codet5_4")
# tokenizer.save_pretrained("./fine_tuned_codet5_4")
#
# from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from datasets import load_dataset
#
# # Încarcă datele de antrenare
# dataset = load_dataset('json', data_files='training_data2.json')
#
# # Împarte dataset-ul în seturi de antrenare și validare
# dataset = dataset['train'].train_test_split(test_size=0.1)
#
# # Încarcă modelul și tokenizer-ul specializat pe cod
# model_name = "Salesforce/codet5-base"  # Model specializat pe cod
# tokenizer = T5Tokenizer.from_pretrained(model_name)  # Tokenizer-ul asociat cu CodeT5
# model = T5ForConditionalGeneration.from_pretrained(model_name)
#
# # Funcție de preprocesare
# def preprocess_function(examples):
#     # Tokenizare cu padding și truncation
#     model_inputs = tokenizer(
#         examples["input"],
#         max_length=512,  # Lungimea maximă acceptată de model
#         padding="max_length",
#         truncation=True
#     )
#     labels = tokenizer(
#         examples["output"],
#         max_length=512,
#         padding="max_length",
#         truncation=True
#     )["input_ids"]
#     model_inputs["labels"] = labels
#     return model_inputs
#
# # Tokenizează dataset-ul
# tokenized_datasets = dataset.map(preprocess_function, batched=True)
#
# # Setări pentru antrenare
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./results",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=3,
#     predict_with_generate=True,
#     fp16=True,  # Folosește FP16 pentru accelerare pe GPU
# )
#
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_datasets["train"],
#     eval_dataset=tokenized_datasets["test"],
#     tokenizer=tokenizer,
# )
#
# # Pornește antrenarea
# trainer.train()
#
# # Salvează modelul antrenat
# model.save_pretrained("./fine_tuned_codet5_4")
# tokenizer.save_pretrained("./fine_tuned_codet5_4")
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainer, Seq2SeqTrainingArguments, \
    RobertaTokenizer
from datasets import load_dataset
import sentencepiece

# Încarcă datele de antrenare
dataset = load_dataset('json', data_files='training_data2.json')

# Împarte dataset-ul în seturi de antrenare și validare
dataset = dataset['train'].train_test_split(test_size=0.1)

# Încarcă modelul și tokenizer-ul
model_name = "google/codet5-base"

# Înlocuiește T5Tokenizer cu RobertaTokenizer
# tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Pre-procesare: tokenizare
def preprocess_function(examples):
    inputs = [ex for ex in examples['input']]
    targets = [ex for ex in examples['output']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(targets, max_length=512, truncation=True).input_ids
    model_inputs["labels"] = labels
    return model_inputs
#
# def preprocess_function(examples):
#     # Tokenizare cu padding și truncation
#     model_inputs = tokenizer(
#         examples["input"],
#         max_length=512,  # Lungimea maximă acceptată de model
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )
#     labels = tokenizer(
#         examples["output"],
#         max_length=512,
#         padding="max_length",
#         truncation=True,
#         return_tensors="pt"
#     )["input_ids"]
#     model_inputs["labels"] = labels
#     return model_inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True)

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
model.save_pretrained("./fine_tuned_codet5_4")
tokenizer.save_pretrained("./fine_tuned_codet5_4")

# ###############





