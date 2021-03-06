import os
import re
import string
from glob import glob
import json
import requests
from zipfile import ZipFile
import sys
import codecs
from datetime import datetime

url_MLQA_V1 = 'https://dl.fbaipublicfiles.com/MLQA/MLQA_V1.zip'

try:
    import git
except:
    pass


def get_root():
    try:
        root = git.Repo(os.getcwd(), search_parent_directories=True).git.rev_parse('--show-toplevel')
    except:
        root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return root

def normalize_text(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def timestamp():
    datetime.now().strftime("%y-%m-%d_%H:%M:%S")