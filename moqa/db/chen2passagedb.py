import sqlite3
import os
import regex as re
from html.parser import HTMLParser
from multiprocessing import Pool
import unicodedata
from tqdm import tqdm
import json
import spacy
import argparse
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

nlp = None
strategy = 'wrap'

PARSER = HTMLParser()
BLACKLIST = { '23443579', '52643645' }  # Conflicting disambig. pages

spacy_lang = {
    "da"   : "da_core_news_sm",
    "nl"   : "nl_core_news_sm",
    "fr"   : "fr_core_news_sm",
    "de"   : "de_core_news_sm",
    "ja"   : "ja_core_news_sm",
    "no"   : "nb_core_news_sm",
    "pl"   : "pl_core_news_sm",
    "pt"   : "pt_core_news_sm",
    "ru"   : "ru_core_news_sm",
    "es"   : "es_core_news_sm",
    # following are processed by multilingual model
    "multi": "xx_ent_wiki_sm",
    }

parser = argparse.ArgumentParser(description='Convert chen_prep.db to passage.db')
parser.add_argument('-l', '--lang', required=True, action='store', type=str)
parser.add_argument('-c', '--chendb', required=True, action='store', type=str)
parser.add_argument('-s', '--data-path', required=True, action='store', type=str)
parser.add_argument('-d', '--dst', required=True, action='store', type=str)
parser.add_argument('-s', '--strategy', default='wrap', required=True, action='store', type=str)
parser.add_argument('--json', action='store_true', type=str)
parser.add_argument('--dpr', action='store_true', type=str)
parser.add_argument('--chen', action='store_true', type=str)
parser.add_argument('--test', action='store_true')


def preprocess(article):
    # Take out HTML escaping WikiExtractor didn't clean
    for k, v in article.items():
        article[k] = PARSER.unescape(v)

    # Filter some disambiguation pages not caught by the WikiExtractor
    if article['id'] in BLACKLIST:
        return None
    if '(disambiguation)' in article['title'].lower():
        return None
    if '(disambiguation page)' in article['title'].lower():
        return None

    # Take out List/Index/Outline pages (mostly links)
    if re.match(r'(List of .+)|(Index of .+)|(Outline of .+)',
                article['title']):
        return None

    # Return doc with `id` set to `title`
    return { 'id': article['title'], 'text': article['text'].replace("Section::::", "") }


def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def normalize(text):
    return unicodedata.normalize('NFD', text)


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global nlp
    global strategy
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            doc = preprocess(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            splits = split_into_100words(doc['text'], nlp, strategy)
            if splits is None:
                continue
            # Add the document
            for split in splits:
                documents.append((normalize(doc['id']), split))
    return documents


def split_into_100words(text, nlp, strategy='wrap'):
    if strategy is None:
        strategy = 'wrap'
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


def from_json(args):
    global nlp
    global strategy

    if args.lang in spacy_lang:
        nlp = spacy.load(spacy_lang[args.lang])
    else:
        nlp = spacy.load(spacy_lang['multi'])
    strategy = args.strategy

    save_path = args.dst

    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    passage_db = conn.cursor()
    passage_db.execute("CREATE TABLE passages (id, lang, title, passage, PRIMARY KEY (id, lang));")
    passage_db.execute("CREATE TABLE extra_documents (id, lang, title, passage, PRIMARY KEY (id, lang));")

    workers = Pool(args.num_workers)

    files = [f for f in iter_files(args.data_path)]
    count = 0
    count_extra = 0
    with tqdm(total=len(files)) as pbar:
        for pairs in tqdm(workers.imap_unordered(get_contents, files)):
            for title, passage in pairs:
                try:
                    passage_db.execute("INSERT INTO passages (id, lang, title, passage) VALUES (?,?,?,?)",
                                       (count, args.lang, title, passage))
                    count += 1
                except Exception as e:
                    print(e)
                    passage_db.execute("INSERT INTO extra_documents (id, lang, title, passage) VALUES (?,?,?,?)",
                                       (count_extra, args.lang, title, passage))
                    count_extra += 1

            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


def from_chen(args):
    if args.lang in spacy_lang:
        nlp = spacy.load(spacy_lang[args.lang])
    else:
        nlp = spacy.load(spacy_lang['multi'])

    conn = sqlite3.connect(args.chendb)
    chendb = conn.cursor()
    len = chendb.execute('SELECT COUNT(*) FROM documents').fetchone()[0]

    dst = os.path.join(args.dst, f"{args.lang}_passage.db")
    if os.path.exists(dst):
        raise RuntimeError(f"DB {dst} already exists")
    print(f"Creating {dst}")
    passage_db = sqlite3.connect(dst)
    passage_db.execute("CREATE TABLE passages (id, lang, title, passage, PRIMARY KEY (id, lang));")

    with tqdm(total=len) as pbar:
        idx = 0
        chendb.execute('SELECT * FROM documents')
        for title, text in chendb:
            splits = split_into_100words(text, nlp, args.strategy)
            if splits is None:
                pbar.update()
                continue
            for split in splits:
                passage_db.execute("INSERT INTO passages (id, lang, title, passage) VALUES (?,?,?,?)",
                                   (idx, args.lang, title, split))
                idx += 1
            pbar.update()

    print("Committing..")
    passage_db.commit()
    passage_db.close()
    conn.close()


def from_dpr(args):
    data_path = 'data/wiki/en/psgs_w100.tsv'
    if args.data_path:
        data_path = args.data_path

    save_path = 'data/wiki/en/en_passage.db'
    if args.dst:
        save_path = args.dst

    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    passage_db = conn.cursor()
    passage_db.execute("CREATE TABLE passages (id, lang, title, passage, PRIMARY KEY (id, lang));")

    count = 0
    with open(data_path) as fp:
        id, text, title = tuple(fp.readline().strip().split('\t'))
        assert id == "id"
        assert text == "text"
        assert title == "title"
        for line in fp:
            count += 1
            id, text, title = tuple(line.strip().split('\t'))
            passage_db.execute("INSERT INTO passages (id, lang, title, passage) VALUES (?,?,?,?)",
                               (id, 'en', title, text[1:-1].replace('""', '"')))
            if count == args.test:
                break
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()


if __name__ == "__main__":
    args = parser.parse_args()
    if args.json:
        from_json(args)
    elif args.chen:
        from_chen(args)
    elif args.dpr:
        from_dpr(args)
    else:
        raise ValueError("Missing option: --json or --chen or --dpr!")
