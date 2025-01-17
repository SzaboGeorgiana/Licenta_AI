# from bs4 import BeautifulSoup
# import requests
#
# # URL-ul paginii
# url = "https://ancabota09.wixsite.com/intern"
# response = requests.get(url)
# soup = BeautifulSoup(response.content, "html.parser")
#
# #cu AI
# # from transformers import pipeline
# #
# # # Creăm pipeline-ul de întrebări și răspunsuri
# # nlp = pipeline("question-answering")
# #
# # # Întrebăm modelul ce element se află asociat cu butonul "contact"
# # result = nlp({
# #     "question": "Care este elementul asociat cu butonul 'contact'?",
# #     "context": str(soup)  # Furnizăm contextul întregii pagini HTML
# # })
# #
# # # Afișăm răspunsul
# # print(result)
# #
#
# #
# # # Funcție pentru construirea XPath-ului
# # def build_xpath(element):
# #     """
# #     Construiește XPath-ul pentru un element dat.
# #     """
# #     path = []
# #     while element.name != '[document]':
# #         siblings = element.find_all_previous(element.name, limit=1)
# #         siblings = [s for s in siblings if s != element]
# #         if siblings:
# #             sibling_index = len(siblings) + 1
# #             path.append(f"{element.name}[{sibling_index}]")
# #         else:
# #             path.append(element.name)
# #         element = element.find_parent()
# #
# #     path.reverse()
# #     return '/' + '/'.join(path)
# # Căutarea butonului cu un text specific
# buttons = soup.find_all(string=lambda text: text and "contact" in text)
# for button in buttons:
#     # print(button)
#     # Obținerea ID-ului butonului (dacă există)
#     if button:
#         button_id = button.find_parent().get("id" )
#         print("ID-ul butonului este:", button_id)
#     else:
#         print("Butonul nu a fost găsit.")
#     # Obținem tag-ul părinte care conține butonul
#
# #
# #     # XPATH
# #     # if button:
# #     #
# #     #     parent = button.find_parent()
# #     #
# #     #     # Construim un XPath de bază pentru buton
# #     #     xpath = build_xpath(parent)
# #     #     print("XPath-ul butonului este:", xpath)
# #     # else:
# #     #     print("Butonul nu a fost găsit.")
from bs4 import BeautifulSoup
import requests

def find_element_with_text(url, search_text):
    # Obținem conținutul paginii
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Căutăm toate elementele care conțin textul dorit
    elements = soup.find_all(string=lambda text: text and search_text.lower() in text.lower())

    # Dacă am găsit elemente
    if elements:
        for element in elements:
            # Găsim părintele elementului (tag-ul care conține textul)
            parent = element.find_parent()

            # Verificăm dacă părintele este un element de tip <script>, <title>, etc.
            if parent and parent.name not in ['script', 'title', 'style']:
                # Verificăm dacă există un ID pentru elementul părinte
                parent_id = parent.get("id")
                if parent_id:
                    return parent_id
    else:
        print(f"Nu am găsit niciun element care conține textul '{search_text}'.")

# Exemplu de utilizare:
url = "https://ancabota09.wixsite.com/intern/contact"
search_text = "contact"  # Poți înlocui cu orice alt text
id=find_element_with_text(url, search_text)
print(f"Am găsit un element cu textul '{search_text}' și ID-ul '{id}'")
