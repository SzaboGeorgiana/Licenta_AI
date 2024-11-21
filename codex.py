# import openai
#
# # Configurare API OpenAI
# openai.api_key = "your_api_key_here"
#
# # Exemplu input/output
# examples = [
#     {
#         "flow": "Open chrome browser\nLoad the page: https://ancabota09.wixsite.com/intern\nValidate that the Book Now button exists\nClick on the Book Now button",
#         "code": "@Test\npublic void verifyBookNowButton() {\n\n    driver.manage().timeouts().implicitlyWait(10, TimeUnit.SECONDS);\n    System.out.println(\"Loaded successfully\");\n\n    driver.get(\"https://ancabota09.wixsite.com/intern\");\n    System.out.println(\"Page is loaded successfully\");\n\n    String search_text = \"BOOK NOW\";\n    WebElement button = driver.findElement(By.cssSelector(\".l7_2fn.wixui-button__label\"));\n\n    if (button.isDisplayed()) {\n        System.out.println(\"The button is displayed\");\n\n        String text = button.getText();\n        Assert.assertEquals(text, search_text, \"Text not correct!\");\n        System.out.println(\"The name of the button is BOOK NOW\");\n\n        button.click();\n\n        String currentUrl = driver.getCurrentUrl();\n        Assert.assertEquals(currentUrl, \"https://ancabota09.wixsite.com/intern/booknow\", \"Bad redirect\");\n        System.out.println(\"The BOOK NOW page is loaded successfully\");\n\n    } else {\n        System.out.println(\"The BOOK NOW button is not displayed\");\n        Assert.fail();\n    }\n}"
#     },
#     # Adaugă alte exemple aici
# ]
#
# # Funcția pentru generarea codului
# def generate_code(flow_description):
#     response = openai.Completion.create(
#         engine="code-davinci-002",  # Folosește Codex
#         prompt=f"Generate Selenium test code for the following flow:\n{flow_description}\n\nCode:\n",
#         temperature=0.2,
#         max_tokens=1500,
#         stop=["\n\n"]
#     )
#     return response.choices[0].text.strip()
#
# # Testare generare cod
# flow = "Open chrome browser\nLoad the page: https://ancabota09.wixsite.com/intern\nValidate that the Book Now button exists\nClick on the Book Now button"
# generated_code = generate_code(flow)
# print("Generated Code:\n", generated_code)