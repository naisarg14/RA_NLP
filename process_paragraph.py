##################################################################################
##################################################################################
##                                                                              ##
##    This file is a part of codes used in the Research Paper titled:           ##
##    "COVID-19 and Digital Health Communication on Rheumatoid Arthritis:       ##
##    A Natural Language Processing Analysis of Reddit Discussion"              ##
##    Copyright (C) 2025  Naisarg Patel (github:naisarg14)                      ##
##                                                                              ##
##    This program is free software: you can redistribute it and/or modify      ##
##    it under the terms of the GNU General Public License as published by      ##
##    the Free Software Foundation, either version 3 of the License, or         ##
##    (at your option) any later version.                                       ##
##                                                                              ##
##    This program is distributed in the hope that it will be useful,           ##
##    but WITHOUT ANY WARRANTY; without even the implied warranty of            ##
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             ##
##    GNU General Public License for more details.                              ##
##                                                                              ##
##    You should have received a copy of the GNU General Public License         ##
##    along with this program.  If not, see <https://www.gnu.org/licenses/>.    ##
##                                                                              ##
##################################################################################
##################################################################################
from nltk.corpus import stopwords
import html
from helpers import get_abbreviations
import re, emoji
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words_all = stopwords.words('english')
remove_stop = ["no", "not", "nor", "none", "neither", "never", "nobody", "nothing", "nowhere", "against", "very"]

stop_words = set([word for word in stop_words_all if word not in remove_stop])

abbreviations = get_abbreviations()

emoticon_dict = {
    ":)": "happy",
    ":(": "sad",
    ":D": "very happy",
    ";)": "wink",
    ":O": "surprised",
    ":|": "neutral",
    ":'(": "crying",
    "XD": "laughing",
    ":/": "skeptical",
    ":3": "cute",
    ">:(": "angry",
    "O:)": "angelic",
    ">:O": "shocked",
    "<3": "love",
    "(:": "happy",
}

lemmatizer = WordNetLemmatizer()

spell = SpellChecker(distance=1)
spell.word_frequency.add("covid")

def correct(word):
    if spell.correction(word) is not None:
        return spell.correction(word)
    else:
        return word

def process_paragraph(paragraph):
    #de-emojinize
    paragraph = emoji.demojize(str(paragraph).strip().lower())
    pattern = r':([a-zA-Z_]+):'
    paragraph = re.sub(pattern, lambda m: m.group(1).replace('_', ' '), paragraph.replace("::", ": :"))

    for emoticon, translation in emoticon_dict.items(): paragraph = paragraph.replace(emoticon, translation)
    #remove links
    paragraph = re.sub(r'http\S+|www\S+|https\S+', '', paragraph, flags=re.MULTILINE)
    #remove emails
    paragraph = re.sub(r'(?:mailto:\s*)?[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', paragraph)
    #remove username
    paragraph = re.sub(r'@\w+', '', paragraph.lower())
    #remove hashtag
    paragraph = re.sub(r'#\w+', '', paragraph.lower())
    #break sentences
    paragraph = paragraph.replace(".", " ")
    #remove html tags
    paragraph = html.unescape(paragraph)
    #keep only text
    paragraph = re.sub(r'[^A-Za-z ]+', '', paragraph)
    #correct stretched words
    paragraph = re.sub(r'(.)\1{2,}', r'\1\1', paragraph).lower()

    tokens = word_tokenize(paragraph)
    for i in range(len(tokens)):
        if tokens[i] in abbreviations:
            tokens[i] = abbreviations[tokens[i]].lower()
    paragraph = " ".join(tokens)


    tokens = [word.lower() for word in word_tokenize(paragraph) if word not in stop_words]
    corrected_tokens = [lemmatizer.lemmatize(correct(word)) for word in tokens]
    correct_paragraph = " ".join(corrected_tokens)

    return correct_paragraph.strip().lower()

if __name__ == "__main__":
    print(f"\n{process_paragraph(input("Text: "))}")