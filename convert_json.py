# Creează o listă de exemple
import pandas as pd
import json

from read_from_excel import flow_list, code_list

examples = []


for flow, code in zip(flow_list, code_list):

    if pd.notna(flow) and pd.notna(code):  # Verifică dacă flow-ul și codul există
        examples.append({
            "input": flow.strip(),
            "output": code.strip()
        })

# Salvează exemplele în format JSON
with open('training_data.json', 'w') as json_file:
    json.dump(examples, json_file, indent=4)

print("Datele pentru antrenare au fost salvate în training_data.json")
