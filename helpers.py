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
import os
from datetime import datetime


def backup(file_path):
    if not os.path.exists(file_path):
        return False
    master_folder, file = os.path.split(os.path.abspath(file_path))
    target_directory = os.path.join(master_folder, 'backups')
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    modified_time = os.path.getmtime(file_path)
    timestamp = datetime.fromtimestamp(modified_time).strftime("%b-%d-%Y_%H.%M")
    name, ext = os.path.splitext(file)
    target_file = os.path.join(target_directory, f'{name}_{timestamp}{ext}')
    os.rename(file_path, target_file)
    return True



def get_abbreviations():
    return {
        "ra": "rheumatoid arthritis",
        "dr": "doctor",
        "dr.": "doctor",
        "doc": "doctor",
        "ive": "i have",
        "id": "i had",
        "dont": "do not",
        "cant": "cannot",
        "ill": "i will",
        "wont": "will not",
        "im": "I am",
        "ive": "I have",
        "isnt": "is not",
        "arent": "are not",
        "wasnt": "was not",
        "werent": "were not",
        "hasnt": "has not",
        "havent": "have not",
        "hadnt": "had not",
        "doesnt": "does not",
        "didnt": "did not",
        "wouldnt": "would not",
        "shouldnt": "should not",
        "couldnt": "could not",
        "mustnt": "must not",
        "mightnt": "might not",
        "neednt": "need not",
        "yall": "you all",
        "youre": "you are",
        "hes": "he is",
        "shes": "she is",
        "theyre": "they are",
        "whos": "who is",
        "whats": "what is",
        "wheres": "where is",
        "heres": "here is",
        "theres": "there is",
        "lets": "let us",
        "thats": "that is",
        "aint": "is not",
        "gonna": "going to",
        "wanna": "want to",
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "lotta": "lot of",
        "lemme": "let me",
        "gimme": "give me",
        "dunno": "do not know",
        "cmon": "come on",
        "nothin": "nothing",
        "somethin": "something",
        "everythin": "everything",
        "tellin": "telling",
        "showin": "showing",
        "goin": "going",
        "doin": "doing",
        "makin": "making",
        "thinkin": "thinking",
        "theyd": "they would",
        "wanna": "want to",
        "gonna": "going to",
        "gotta": "got to",
        "kinda": "kind of",
        "sorta": "sort of",
        "ain't": "is not",
        "y'all": "you all",
        "cuz": "because",
        "outta": "out of",
        "coulda": "could have",
        "shoulda": "should have",
        "woulda": "would have",
        "hafta": "have to",
        "tryna": "trying to",
        "betcha": "bet you",
        "whatcha": "what are you",
        "bro": "brother",
        "sis": "sister",
        "brb": "be right back",
        "btw": "by the way",
        "lol": "laugh out loud",
        "idk": "I do not know",
        "omg": "oh my god",
        "thx": "thanks",
        "pls": "please",
        "b4": "before",
        "u": "you",
        "r": "are",
        "ur": "your",
        "gr8": "great",
        "l8r": "later",
        "b/c": "because",
        "bday": "birthday",
        "msg": "message",
        "np": "no problem",
        "fyi": "for your information",
        "tbh": "to be honest",
        "rn": "right now",
        "tho": "though",
        "bff": "best friend forever",
        "omw": "on my way",
        "bc": "because",
        "tks": "thanks",
        "thnks": "thanks",
        "w/": "with",
        "w/o": "without",
        "b4n": "bye for now",
        "cya": "see you",
        "gr8": "great",
        "lmk": "let me know",
        "smh": "shaking my head",
        "tbh": "to be honest",
        "ikr": "I know right",
        "rofl": "rolling on the floor laughing",
        "np": "no problem",
        "imo": "in my opinion",
        "fomo": "fear of missing out",
        "irl": "in real life",
        "afk": "away from keyboard",
        "gg": "good game",
        "yw": "you're welcome",
        "atm": "at the moment",
        "bbl": "be back later",
        "bfn": "bye for now",
        "cu": "see you",
        "ez": "easy",
        "hbu": "how about you",
        "hbd": "happy birthday",
        "hmu": "hit me up",
        "jk": "just kidding",
        "nvm": "never mind",
        "oic": "oh I see",
        "omg": "oh my god",
        "sup": "what's up",
        "ttyl": "talk to you later",
        "txt": "text",
        "wtf": "what the heck",
        "yolo": "you only live once",
        "ttys": "talk to you soon",
        "y": "why",
        "yrs": "years",
        "yr": "year",
        "u": "you",
        "youll": "you will",
        'lmao': "laughing",
        'lmfao': "laughing",
        'meds': "medications",
    }
