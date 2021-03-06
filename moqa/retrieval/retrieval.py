import os
from moqa.common.utils import get_root, timestamp
from moqa.db.db import PassageDB
from typing import AnyStr
from argparse import Namespace

# lucene imports
import lucene
from java.nio.file import Paths
# from org.apache.lucene.analysis.miscellaneous import LimitTokenCountAnalyzer
from org.apache.lucene.document import Document, Field, FieldType
from org.apache.lucene.index import \
    FieldInfo, IndexWriter, IndexWriterConfig, IndexOptions
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.search import IndexSearcher

from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.de import GermanAnalyzer
from org.apache.lucene.analysis.es import SpanishAnalyzer
from org.apache.lucene.analysis.ar import ArabicAnalyzer
from org.apache.lucene.analysis.ru import RussianAnalyzer
from org.apache.lucene.analysis.cjk import CJKAnalyzer
from org.apache.lucene.analysis.fi import FinnishAnalyzer
from org.apache.lucene.analysis.th import ThaiAnalyzer
from org.apache.lucene.analysis.tr import TurkishAnalyzer
from org.apache.lucene.analysis.sv import SwedishAnalyzer
from org.apache.lucene.analysis.br import BrazilianAnalyzer
from org.apache.lucene.analysis.pt import PortugueseAnalyzer
from org.apache.lucene.analysis.no import NorwegianAnalyzer
from org.apache.lucene.analysis.it import ItalianAnalyzer
from org.apache.lucene.analysis.id import IndonesianAnalyzer
from org.apache.lucene.analysis.hu import HungarianAnalyzer
from org.apache.lucene.analysis.pl import PolishAnalyzer
from org.apache.lucene.analysis.fr import FrenchAnalyzer
from org.apache.lucene.analysis.nl import DutchAnalyzer
from org.apache.lucene.analysis.da import DanishAnalyzer
from org.apache.lucene.analysis.hi import HindiAnalyzer
from org.apache.lucene.analysis.bn import BengaliAnalyzer
from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer

analyzers = {
    'standard': StandardAnalyzer,
#    'vi': StandardAnalyzer,     # Vietnamese
#    'te': StandardAnalyzer,     # Telugu
#    'sw': StandardAnalyzer,     # Swahili
#    'ms': StandardAnalyzer,     # Malay
#    'km': StandardAnalyzer,     # Khmer
#    'he': StandardAnalyzer,     # Hebrew

    'en': EnglishAnalyzer,
    'es': SpanishAnalyzer,
    'de': GermanAnalyzer,
    'ar': ArabicAnalyzer,
    'ru': RussianAnalyzer,
    'ko': CJKAnalyzer, # Korean
    'ja': CJKAnalyzer, # Japanese
    'fi': FinnishAnalyzer,
    'th': ThaiAnalyzer,
    'tr': TurkishAnalyzer,
    'sv': SwedishAnalyzer,
    'br': BrazilianAnalyzer,
    'pt': PortugueseAnalyzer,
    'no': NorwegianAnalyzer,
    'it': ItalianAnalyzer,
    'id': IndonesianAnalyzer,
    'hu': HungarianAnalyzer,
    'pl': PolishAnalyzer,
    'fr': FrenchAnalyzer,
    'nl': DutchAnalyzer,
    'da': DanishAnalyzer,
    'hi': HindiAnalyzer,
    'bn': BengaliAnalyzer,
    'zh': SmartChineseAnalyzer,
    }

class Retriever(object):
    def __init__(self, k1=None, b=None):
        if k1 is None:
            self.k1=1.8
        else:
            self.k1=k1

        if b == None:
            self.b=0.1
        else:
            self.b=b

    @classmethod
    def get_index_name(self, lang: AnyStr, db_path: AnyStr, index_path: AnyStr, suffix=""):
        db_name = os.path.basename(db_path)
        if suffix != "":
            name = f"{db_name}_{lang}_{suffix}.index"
        else:
            name = f"{db_name}_{lang}.index"
        path = os.path.join(get_root(),'data', 'indexes') if index_path is None else index_path
        return os.path.join(path, name)

class Indexer(Retriever):
    def __init__(self, lang: AnyStr, db: PassageDB, analyzer: AnyStr, index_dir=None, ram_size=2048 ):
        """ Returns scored documents in multiple languages.

        Parameters:
        dataset  (str): ['mlqa_dev', 'mlqa_test', 'wiki']
        lang     (str): ['en', 'es', 'de']
        anlyzer  (str): ['en', 'es', 'de', 'standard']
        ram_size (int): Size of memory used while indexing

        Returns:
        """
        super().__init__()

        idxdir = self.get_index_name(lang, db.path, index_dir)
        self.db = db

        # stores index files, poor concurency try NIOFSDirectory instead
        store = SimpleFSDirectory(Paths.get(idxdir))
        # limit max. number of tokens per document.
        # analyzer will not consume more tokens than that
        #analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
        # configuration for index writer
        config = IndexWriterConfig(analyzers[analyzer]())
        # creates or overwrites index
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        # setting similarity BM25Similarity(k1=1.2,b=0.75)
        similarity = BM25Similarity(self.k1, self.b)
        config.setSimilarity(similarity)
        config.setRAMBufferSizeMB(float(ram_size))
        # create index writer
        self.writer = IndexWriter(store, config)

        self.ftdata = FieldType()
        self.ftmeta = FieldType()
        # IndexSearcher will return value of the field
        self.ftdata.setStored(False)
        self.ftmeta.setStored(True)
        # will be analyzed by Analyzer
        self.ftdata.setTokenized(True)
        self.ftmeta.setTokenized(False)
        # what information are stored (probably DOCS would be sufficient)
        # DOCS: Only documents are indexed: term frequencies and positions are omitted.
        #       Phrase and other positional queries on the field will throw an exception,
        #       and scoring will behave as if any term in the document appears only once.
        # DOCS_AND_FREQS: Only documents and term frequencies are indexed: positions are
        #       omitted. This enables normal scoring, except Phrase and other positional
        #       queries will throw an exception.
        # DOCS_AND_FREQS_AND_POSITIONS: Indexes documents, frequencies and positions.
        #       This is a typical default for full-text search: full scoring is enabled
        #       and positional queries are supported.
        self.ftdata.setIndexOptions(IndexOptions.DOCS_AND_FREQS)
        self.ftmeta.setIndexOptions(IndexOptions.DOCS)
        # instantiate some reusable objects
        self.doc = Document()
        # Id cannot be reused because there is multiple values
        # I could store list of fields and add one if its not enough
        #self.fieldId = Field("id", "dummy", self.ftmeta)
        self.fieldTitle = Field("title", "dummy", self.ftdata)
        self.doc.add(self.fieldTitle)
        self.fieldContext = Field("context", "dummy", self.ftdata)
        self.doc.add(self.fieldContext)
        self.fieldId    = Field("id", "dummy", self.ftmeta)
        self.fieldIdx     = Field("idx", "dummy", self.ftmeta)

    def createIndex(self):
        for i, doc in enumerate(self.db):
            self.fieldIdx.setIntValue(i)
            self.fieldId.setStringValue(doc['id'])
            self.fieldTitle.setStringValue(doc['title'])
            self.fieldContext.setStringValue(doc['context'])
            self.writer.addDocument(self.doc)
        self.commit()

    def commit(self):
        self.writer.commit()
        self.writer.close()

class Searcher(Retriever):
    def __init__(self, k1=None, b=None):
        super().__init__(k1, b)
        print("Searcher k1: {}, b: {}", self.k1, self.b)
        self.similarity = BM25Similarity(self.k1, self.b)
        self.searcher = {}
        self.parser = {}
        self.multi_parser = {}
        self.__call__ = self.query

    def addLang(self, lang, analyzer, index_name, index_path=None):
        """ Initialises index searcher in different languages

        Parameters:
        lang     (str): ['en', 'es', 'de']
        anlyzer  (str): ['en', 'es', 'de', 'standard']
        index_path (str): path to index

        Returns:
        """
        idxdir = self.get_index_name(lang=lang, db_path=index_name, index_path=index_path)
        if not os.path.exists(idxdir):
            raise ValueError(f"No index in {idxdir}!")
        directory = SimpleFSDirectory(Paths.get(idxdir))
        self.searcher[lang] = IndexSearcher(DirectoryReader.open(directory))
        self.parser[lang]   = QueryParser("context", analyzers[analyzer]())
        self.multi_parser[lang] = MultiFieldQueryParser(["context", "title"], analyzers[analyzer])

        self.searcher[lang].setSimilarity(self.similarity)

    def printResult(self, scoreDocs, lang):
        print("Number of retrieved documents:", len(scoreDocs))
        for scoreDoc in scoreDocs:
            doc = self.searcher[lang].doc(scoreDoc.doc)
            print("Score:", scoreDoc.score)
            self.printDoc(doc)

    def printDoc(self, doc):
        if type(doc) is Namespace:
            print("Language:", doc.lang)
            print("Score:", doc.score)
            doc = doc.doc
        print("Id:", doc.get('id').stringValue())
        print("Index:", doc.get('idx').stringValue())
        print("Name:", doc.get("title").encode('utf-8'))
        print("Context:", doc.get("context").encode('utf-8'))
        print("------------------------------------------------------")

    def getDoc(self, scoreDoc, lang):
        return self.searcher[lang].doc(scoreDoc.doc)

    def queryTest(self, command, lang):
        q = self.query(command, lang, 5)
        self.printResult(q, lang)
        return q

    def query(self, command, lang, topk, field='context'):
        """
        Retrieve documents for question
        """
        if not field == 'context' or not field == 'context_title':
            raise ValueError(f"Cannot search by {field}!")
        if lang not in self.searcher:
            raise RuntimeError(f"Language '{lang}' not added")

        esccommand = self.parser[lang].escape(command)
        if field == 'context':
            query = self.parser[lang].parse(esccommand)
        else:
            query = self.multi_parser[lang].parse(esccommand)
        scoreDocs = self.searcher[lang].search(query, topk).scoreDocs

        docs = []
        for scoreDoc in scoreDocs:
            docs.append(Namespace(score=scoreDoc.score, doc=self.getDoc(scoreDoc), lang=lang))
        return docs

    #def queryMulti(self, command, lang, n=50, p=1):
    #    """ Returns scored documents in multiple languages.

    #    Parameters:
    #    command (str): query string
    #    lang    (str): language in which is the query
    #    n       (int): number of documents retrieved
    #    p       (float): reduces number of retrieved documents from each language
    #                     e.g.: for 3 languages, n = 50 and p = 0.5 from each language
    #                     25 documents will be retrieved.
    #                     Must satisfy n*len(langs)*p >= n

    #    Returns:

    #    [scoreDocs]: ordered list of scored documents by their score

    #    """
    #    scoreDocs = []
    #    for to in self.languages:
    #        transl_comm = self.translator(lang, to, command)
    #        scoreDocs.append(self.query(transl_comm, to, int(n*p)))
    #    return scoreDocs.sort(key=lambda x: x.score, reverse=True)[:n]