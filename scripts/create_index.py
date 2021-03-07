from moqa.retrieval.retrieval import Indexer
from moqa.db.db import PassageDB
import os
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--db-path', default="", type=str)
parser.add_argument('--lang', required=True, type=str)

def create_index(args):
    if args.db_path:
        db_path = args.db_path
    else:
        db_path = os.path.join('data', 'wiki', args.lang, f"{args.lang}_passage.db")
    logging.info(f"Creating index from {db_path}")
    with PassageDB(db_path) as db:
        indexer = Indexer(args.lang, db, args.lang, index_dir=args.index_dir, ram_size=8*1024)
        indexer.createIndex()

if __name__ == "__main__":
    args = parser.parse_args()
    create_index(args)