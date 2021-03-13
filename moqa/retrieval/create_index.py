from moqa.retrieval import Indexer, Searcher
from moqa.db import PassageDB
from moqa.common import config
import os
import argparse
import logging

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=config.log_file,
    level=config.log_level)

parser = argparse.ArgumentParser()
parser.add_argument('--db-path', default="", type=str)
parser.add_argument('--lang', required=True, type=str)
parser.add_argument('--index_dir', default=None, type=str)
parser.add_argument('--test', action='store_true')


def create_index(args):
    if args.db_path:
        db_path = args.db_path
    elif args.lang:
        db_path = os.path.join('data', 'wiki', args.lang, f"{args.lang}_passage.db")
    else:
        parser.print_help()
        raise RuntimeError("Either --db-path or --lang must be set!")
    logging.info(f"Creating index from {db_path}")
    with PassageDB(db_path) as db:
        indexer = Indexer(args.lang, db, args.lang, index_dir=args.index_dir, ram_size=8 * 1024)
        indexer.createIndex()
    logging.info(f"Index for {db.path} was created in {indexer.idx_dir}!")


def test_index(args):
    if args.db_path:
        db_path = args.db_path
    elif args.lang:
        db_path = os.path.join('data', 'wiki', args.lang, f"{args.lang}_passage.db")
    else:
        parser.print_help()
        raise RuntimeError("Either --db-path or --lang must be set!")
    with PassageDB(db_path) as db:
        searcher = Searcher()
        searcher.addLang(args.lang, db)
        query = "i dansk og skandinavisk"
        docs = searcher.query(query, args.lang, topk=5, field='context_title')
        for doc in docs:
            searcher.printDoc(doc)
            passage, title, id = db.get_doc_text(doc.id, ['passage', 'title', 'id'])
            print("id from db", id)
            print("id from index", doc.id)
            print("passage", passage)
            print("title", title)


if __name__ == "__main__":
    arguments = parser.parse_args()
    print(arguments)
    if arguments.test:
        test_index(arguments)
    else:
        create_index(arguments)
