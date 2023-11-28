import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# nltk.download('stopwords')
# nltk.download('punkt')

import os
import pandas as pd
import docx2txt
from pdfminer.high_level import extract_text
import re
import spacy
from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_web_sm

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

PHONE_REG = re.compile(r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]')
EMAIL_REG = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
ADDRESS_REG = re.compile(r'\b\d{1,5}\s\w+\s\w+,\s\w+\s\d{5}\b|\b\d{1,5}\s\w+\s\w+,\s\w+\b')

def extract_text_from_docx(docx_path):
    txt = docx2txt.process(docx_path)
    if txt:
        txt = txt.replace('\n', ' ')
        return txt.replace('\t', ' ')
    return None

def extract_text_from_pdf(pdf_path):
    txt = extract_text(pdf_path)
    if txt:
        return  txt.replace('\t', ' ')
    return None

# https://spacy.io/usage/rule-based-matching
def extract_name(resume_text):
    nlp_text = nlp(resume_text)
    pattern = [{'POS': 'PROPN'}, {'POS': 'PROPN'}]
    matcher = Matcher(nlp.vocab)
    matcher.add('NAME', [pattern])
    matches = matcher(nlp_text)
    for match_id, start, end in matches:
        span = nlp_text[start:end]
        matcher.remove('NAME')
        return span.text

def extract_education(resume_text):
    EDUCATION = ['BE', 'B.E.', 'B.E', 'BS', 'B.S', 'ME', 'M.E', 'M.E.', 'MS', 'M.S',
                 'BTECH', 'B.TECH', 'M.TECH', 'MTECH', 'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII']

    stop_words = set(stopwords.words('english'))
    words = word_tokenize(resume_text)
    cleaned_words = []
    for word in words:
        if word.isalnum() not in stop_words:
            cleaned_words.append(word)
    education = []
    for word in cleaned_words:
        if word in EDUCATION:
            education.append(word.upper())

    return education

def extract_contact_info(resume_text):
    phone = re.findall(PHONE_REG, resume_text)
    email = re.findall(EMAIL_REG, resume_text)
    address = re.findall(ADDRESS_REG, resume_text)

    phone_number = ''.join(phone[0]) if phone else None
    email_address = email[0] if email else None
    address_info = address[0].strip() if address else None
    return phone_number, email_address, address_info

def process_resumes(folder_path):
    data = {'Names': [], 'Phone Number': [], 'Email': [], 'Address': [], 'Education': []}

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.endswith(".docx"):
            text = extract_text_from_docx(file_path)
        elif filename.endswith(".pdf"):
            text = extract_text_from_pdf(file_path)
        else:
            continue

        names = extract_name(text)
        phone_number, email_address, address_info = extract_contact_info(text)
        education_info = extract_education(text)

        data['Names'].append(names)
        data['Phone Number'].append(phone_number)
        data['Email'].append(email_address)
        data['Address'].append(address_info)
        data['Education'].append(education_info)

    df = pd.DataFrame(data)
    return df

if __name__ == '__main__':
    folder_path = 'resumes'
    resume_df = process_resumes(folder_path)
    # print(resume_df)
    print(resume_df.to_string(index=False)) # to hide index
