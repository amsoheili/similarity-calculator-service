# import argostranslate.package
# import argostranslate.translate
# import logging
#
# logging.basicConfig(level=logging.DEBUG)
#
# from_code = "fa"
# to_code = "en"
# # Download and install Argos Translate package
# # argostranslate.package.update_package_index()
# # available_packages = argostranslate.package.get_available_packages()
# # package_to_install = next(
# #     filter(
# #         lambda x: x.from_code == from_code and x.to_code == to_code, available_packages
# #     )
# # )
# # argostranslate.package.install_from_path(package_to_install.download())
#
# # Translate
# translatedText = argostranslate.translate.translate("ما به دنبال یک توسعه‌دهنده فرانت‌اند حرفه‌ای هستیم تا در توسعه و بهینه‌سازی یک اپلیکیشن بانکی جدید و یک پلتفرم صرافی دیجیتال با تیم ما همکاری کند. این نقش فرصت منحصربه‌فردی را برای کار بر روی پروژه‌های داخلی و بین‌المللی فراهم می‌کند، از جمله همکاری با تیم‌هایی از ایالات متحده و اروپا در توسعه محصولات دیجیتال مدرن.", from_code, to_code)
# print(translatedText)
# # '¡Hola Mundo!'


# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"


# model = AutoModelForCausalLM.from_pretrained(
#     "universitytehran/PersianMind-v1.0",
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     device_map={"": device},
# )

# tokenizer = AutoTokenizer.from_pretrained("universitytehran/PersianMind-v1.0")

# TEMPLATE = "{context}\nYou: {prompt}\nPersianMind: "
# CONTEXT = "This is a conversation with PersianMind. It is an artificial intelligence model designed by a team of NLP experts at the University of Tehran to help you with various tasks such as answering questions, providing recommendations, and helping with decision making. You can ask it anything you want and it will do its best to give you accurate and relevant information."
# PROMPT = "در مورد هوش مصنوعی توضیح بده."

# model_input = TEMPLATE.format(context=CONTEXT, prompt=PROMPT)
# input_tokens = tokenizer(model_input, return_tensors="pt").to(device)
# generate_ids = model.generate(**input_tokens, max_new_tokens=512, do_sample=True, repetition_penalty=1.1)
# model_output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
# print('here')
# print(model_output[len(model_input):])

# import torch
# from transformers import AutoTokenizer, AutoModel

# model_name = "universitytehran/PersianMind-v1.0"

# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)
# model.eval()  # IMPORTANT: set to evaluation mode

# text = "سلام دنیا"  # "Hello world" in Persian

# inputs = tokenizer(text, return_tensors="pt")

# with torch.no_grad():  # Save memory here
#     outputs = model(**inputs)
#     last_hidden_state = outputs.last_hidden_state

# print(last_hidden_state.shape)


# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# # Step 1: Load model and tokenizer
# model_name = "facebook/m2m100_418M"
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# # Step 2: Your Persian text
# text = "سلام، حال شما چطور است؟"  # "Hello, how are you?"

# # Step 3: Set source language (Persian = 'fa')
# tokenizer.src_lang = "fa"

# # Step 4: Tokenize input
# encoded_input = tokenizer(text, return_tensors="pt")

# # Step 5: Generate translation
# with torch.no_grad():
#     generated_tokens = model.generate(
#         **encoded_input,
#         forced_bos_token_id=tokenizer.get_lang_id("en"),  # Target language = English
#         max_length=100
#     )

# # Step 6: Decode output
# translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

# # Step 7: Print
# print("Original (Persian):", text)
# print("Translated (English):", translated_text)

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Specify the model name
model_name = "universitytehran/PersianMind-v1.0"

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def translate_persian_to_english(persian_text):
    """
    Translates Persian text to English using the PersianMind model.

    Args:
        persian_text (str): The Persian text to translate.

    Returns:
        str: The translated English text.
    """
    try:
        # Tokenize the input Persian text
        inputs = tokenizer(persian_text, return_tensors="pt", padding=True, truncation=True)

        # Generate the translation
        outputs = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)

        # Decode the generated output to English text
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return translated_text
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return None

# Example usage:
persian_input = "سلام، حال شما چطور است؟"
english_translation = translate_persian_to_english(persian_input)

if english_translation:
    print(f"Persian Text: {persian_input}")
    print(f"English Translation: {english_translation}")

persian_input_2 = "هوا امروز خیلی گرم است."
english_translation_2 = translate_persian_to_english(persian_input_2)

if english_translation_2:
    print(f"\nPersian Text: {persian_input_2}")
    print(f"English Translation: {english_translation_2}")

# You can try with more complex sentences
persian_input_3 = "دانشجویان در حال مطالعه برای امتحان نهایی هستند."
english_translation_3 = translate_persian_to_english(persian_input_3)

if english_translation_3:
    print(f"\nPersian Text: {persian_input_3}")
    print(f"English Translation: {english_translation_3}")
