import pandas as pd
# Creează o listă de exemple
import json
# Citește fișierul Excel
file_path = '../baze_de_date/excel/set_chat.xlsx'
df = pd.read_excel(file_path)

# Extrage coloanele care conțin "flow-ul" și "codul"
flow_column = df['Actions']
code_column = df['Code']

code_list = []
flow_list=[]
string1=""

# Parcurge fiecare cod și adaugă-l în listă dacă nu este NaN
for flow, code in zip(flow_column, code_column):
    if pd.notna(code):  # Verifică dacă valoarea nu este NaN
        code_list.append(str(code))  # Adaugă codul convertit în string în listă
        if pd.notna(flow):  # Verifică dacă valoarea nu este NaN
            string1 = flow
            string1 += "\n"
    else:
        if pd.notna(flow):  # Verifică dacă valoarea nu este NaN
            string1 += flow
            string1 += "\n"
        else:
            if string1 != "":
                flow_list.append(str(string1))
flow_list.append(str(string1))

i = 0
for flow, code in zip(flow_list, code_list):
    i+=1
    print(i)
    print("\nFlow:\n"+flow+"\nCod:\n"+code+"\n")

#
# examples = []
#
# for flow, code in zip(flow_list, code_list):
#
#     if pd.notna(flow) and pd.notna(code):  # Verifică dacă flow-ul și codul există
#         examples.append({
#             "input": flow.strip(),
#             "output": code.strip()
#         })
#
# # Salvează exemplele în format JSON
# with open('../baze_de_date/json/training_data2.json', 'w') as json_file:
#     json.dump(examples, json_file, indent=4)
#
# print("Datele pentru antrenare au fost salvate în training_data.json")
