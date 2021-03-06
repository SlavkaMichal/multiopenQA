"""
author: Martin Fajcik, drqa's authors

"""
import sqlite3
from typing import AnyStr


class PassageDB:
    def __init__(self, db_path: AnyStr):
        self.path = db_path
        self.connection = sqlite3.connect(db_path, check_same_thread=False)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        return self.path

    def close(self):
        self.connection.close()

    def get_doc_text(self, doc_id, columns="passage"):
        """

        :param doc_id:
        :param columns:
        :return:
        """
        if type(columns) == list:
            columns = ", ".join(columns)

        cursor = self.connection.cursor()
        cursor.execute(f"SELECT {columns} FROM paragraphs WHERE id = ?", (doc_id,))
        result = cursor.fetchone()
        cursor.close()
        if result is  None:
            raise ValueError(f"ID {doc_id} not in the database!")
        return result

    def get_doc_ids(self, table="paragraphs"):
        """Fetch all ids of docs stored in the db."""
        cursor = self.connection.cursor()
        cursor.execute(f"SELECT id FROM {table}")
        results = [r[0] for r in cursor.fetchall()]
        cursor.close()
        return results

