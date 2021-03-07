import sqlite3
import os
from multiprocessing import Pool
from tqdm import tqdm
import spacy
import argparse

spacy_lang = {
    "da": "da_core_news_sm",
    "nl": "nl_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "ja": "ja_core_news_sm",
    "no": "nb_core_news_sm",
    "pl": "pl_core_news_sm",
    "pt": "pt_core_news_sm",
    "ru": "ru_core_news_sm",
    "es": "es_core_news_sm",
    }

parser = argparse.ArgumentParser(description='Convert chen_prep.db to passage.db')
parser.add_argument('-l', '--lang', required=True, action='store', type=str)
parser.add_argument('-c', '--chendb', required=True, action='store', type=str)
parser.add_argument('-d', '--dst', required=True, action='store', type=str)
parser.add_argument('-s', '--strategy', required=True, action='store', type=str)
parser.add_argument('--test', action='store_true')


def split_into_100words(text, nlp, strategy='wrap'):
    processed = nlp(text.replace('\n\n', '\n'))
    if len(processed) < 50:
        # its likely this is disambiguation page
        return None

    splits = []
    start = processed[:100]
    while len(processed) >= 100:
        splits.append(''.join(t.text_with_ws for t in processed[:100]))
        processed = processed[100:]

    if len(processed) < 10:
        # if its too small throw the remainder away
        pass
    elif strategy == 'wrap':
        # wrap around start and end to make 100
        splits.append(
            ''.join(t.text_with_ws for t in processed) +
            ''.join(t.text_with_ws for t in start[:100 - len(processed)])
            )
    elif strategy == 'no_wrap':
        # add only the reminder
        splits.append(''.join(t.text_with_ws for t in processed))
    elif strategy == 'throw_out':
        pass
    else:
        raise ValueError(f"Strategy {strategy} is not supported.")

    return splits


def main():
    args = parser.parse_args()
    nlp = spacy.load(spacy_lang[args.lang])

    conn = sqlite3.connect(args.chendb)
    chendb = conn.cursor()
    len = chendb.execute('SELECT COUNT(*) FROM documents').fetchone()[0]

    dst = os.path.join(args.dst, f"{args.lang}_passage.db")
    if os.path.exists(dst):
        raise RuntimeError(f"DB {dst} already exists")
    print(f"Creating {dst}")
    passage_db = sqlite3.connect(dst)
    passage_db.execute("CREATE TABLE passages (id PRIMARY KEY, title, passage);")

    with tqdm(total=len) as pbar:
        idx = 0
        chendb.execute('SELECT * FROM documents')
        for title, text in chendb:
            splits = split_into_100words(text.replace("Section::::",""), nlp, args.strategy)
            if splits is None:
                pbar.update()
                continue
            for split in splits:
                passage_db.execute("INSERT INTO passages VALUES (?,?,?)", (idx, title, split))
                idx += 1
            pbar.update()

    print("Committing..")
    passage_db.commit()
    passage_db.close()
    conn.close()


if __name__ == "__main__":
    main()
