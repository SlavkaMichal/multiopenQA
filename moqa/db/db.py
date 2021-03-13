"""
author: Martin Fajcik, drqa's authors

"""
import sqlite3
from typing import AnyStr
import os

class PassageDB:
    def __init__(self, db_path: AnyStr):
        if not os.path.isfile(db_path):
            print(os.getcwd())
            raise RuntimeError(f"Database file {db_path} does not exists!")
        self.path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)
        self._len = -1

    def __del__(self):
        # just in case
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __len__(self):
        if self._len == -1:
            cursor = self.connection.cursor()
            self._len = cursor.execute('SELECT COUNT(*) FROM passages').fetchone()[0]

        return self._len

    def __iter__(self):
        iterator = self.connection.cursor()
        iterator.execute('SELECT id, title, passage FROM passages')
        for id, title, passage in iterator:
            yield (id, title, passage)

    def path(self):
        return self.path

    def close(self):
        self.connection.close()

    def get_doc_text(self, doc_id, lang=None, columns="passage"):
        """

        :param doc_id:
        :param columns:
        :return:
        """
        if type(columns) == list:
            columns = ", ".join(columns)

        cursor = self.connection.cursor()
        if lang is None:
            cursor.execute(f"SELECT {columns} FROM passages WHERE id = ?", (doc_id,))
        else:
            cursor.execute(f"SELECT {columns} FROM passages WHERE id = ? AND lang = ?", (doc_id, lang))

        result = cursor.fetchone()
        cursor.close()
        if result is None and lang is None:
            raise ValueError(f"ID {doc_id} not in the database!")
        elif result is None:
            raise ValueError(f"ID {doc_id} and {lang} not in the database!")
        return result

    def get_doc_ids(self, table="passages"):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT id FROM {table}")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results
