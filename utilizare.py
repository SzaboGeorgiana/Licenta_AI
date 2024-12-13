from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer

# Încarcă modelul antrenat
model_path = "./fine_tuned_codet5_3"
# tokenizer = T5Tokenizer.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")

model = T5ForConditionalGeneration.from_pretrained(model_path)

# Prompt-ul pentru generare
# input_text = "Click on search button"
# input_text = "Open chrome browser\nLoad the page: https://example.com"
# \nCheck if the login button exists\nClick the login button.\nCheck if url is \"emag.ro\"

# input_text = "Verify that clicking on \"View Cart\" displays the correct items in the cart"
# input_text = "Verify the \"Apply Now\" button redirects to the \"https://example.com/job-application\" page"
# input_text="Wait for the text \"Successfully logged out\" to appear after clicking logout"
# input_text="Wait for the text \"hello\" in to appear"
# input_text="Verify the \"Add to Wishlist\" button is visible and has correct text"
# input_text="Verify the \"Add to Wishlist\" button has correct text"
input_text="Write in Adults number field \"3\""

input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generare cod
outputs = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

#
# import csv
# from transformers import T5Tokenizer, T5ForConditionalGeneration, RobertaTokenizer
#
# # Încarcă modelul și tokenizer-ul
# model_path = "./fine_tuned_codet5_3"  # Adaptează calea dacă este diferită
# tokenizer = RobertaTokenizer.from_pretrained("Salesforce/codet5-base")
# model = T5ForConditionalGeneration.from_pretrained(model_path)
#
# # Datele de intrare
# instructions = [
#     "Open chrome browser",
#     "Load the page: https://ancabota09.wixsite.com/intern",
#     "Validate that the Contact mail button exists",
#     "Click on mail address",
#     "",
#     "Open chrome browser",
#     "Load the page: https://ancabota09.wixsite.com/intern",
#     "Validate that the Chat icon exists",
#     "Click on Chat icon",
#     "",
#     "Open chrome browser",
#     "Load the page: https://ancabota09.wixsite.com/intern (implicit Home page)",
#     "Click on the Rooms button",
#     "Click on the More Info button",
#     "Validate that the Book now button exists",
#     "Complete Check In date",
#     "Complete Check Out date",
#     "Complete Adults number",
#     "Click on the Book now button"
# ]
#
# # Filtrăm liniile goale pentru a evita input-uri inutile
# instructions = [inst for inst in instructions if inst.strip()]
#
# # Creăm un fișier CSV pentru rezultate
# output_file = "test_results.csv"
#
#
# # Funcție pentru generare cod
# def generate_code(instruction):
#     input_ids = tokenizer.encode(instruction, return_tensors="pt")
#     outputs = model.generate(input_ids, max_length=200, num_beams=4, early_stopping=True)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
#
#
# # Procesăm fiecare instrucțiune și salvăm rezultatele
# with open(output_file, mode="w", newline="", encoding="utf-8") as csv_file:
#     writer = csv.writer(csv_file)
#     writer.writerow(["Instruction", "Generated Code"])  # Header
#
#     for instruction in instructions:
#         try:
#             code = generate_code(instruction)
#             writer.writerow([instruction, code])
#             print(f"Processed: {instruction}")
#         except Exception as e:
#             writer.writerow([instruction, f"Error: {str(e)}"])
#             print(f"Error processing: {instruction} - {str(e)}")
#
# print(f"Rezultatele au fost salvate în {output_file}.")
