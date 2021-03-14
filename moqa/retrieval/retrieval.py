import os
from moqa.db.db import PassageDB
from typing import AnyStr
from argparse import Namespace
from tqdm import tqdm

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
from org.apache.lucene.search import BooleanClause
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
# from org.apache.lucene.analysis.cn.smart import SmartChineseAnalyzer
from moqa.common import config
import logging

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=config.log_file,
    level=config.log_level)

analyzers = {
    'standard': StandardAnalyzer,
    #    'vi': StandardAnalyzer,     # Vietnamese
    #    'te': StandardAnalyzer,     # Telugu
    #    'sw': StandardAnalyzer,     # Swahili
    #    'ms': StandardAnalyzer,     # Malay
    #    'km': StandardAnalyzer,     # Khmer
    #    'he': StandardAnalyzer,     # Hebrew

    'en'      : EnglishAnalyzer,
    'es'      : SpanishAnalyzer,
    'de'      : GermanAnalyzer,
    'ar'      : ArabicAnalyzer,
    'ru'      : RussianAnalyzer,
    'ko'      : CJKAnalyzer,  # Korean
    'ja'      : CJKAnalyzer,  # Japanese
    'fi'      : FinnishAnalyzer,
    'th'      : ThaiAnalyzer,
    'tr'      : TurkishAnalyzer,
    'sv'      : SwedishAnalyzer,
    'br'      : BrazilianAnalyzer,
    'pt'      : PortugueseAnalyzer,
    'no'      : NorwegianAnalyzer,
    'it'      : ItalianAnalyzer,
    'id'      : IndonesianAnalyzer,
    'hu'      : HungarianAnalyzer,
    'pl'      : PolishAnalyzer,
    'fr'      : FrenchAnalyzer,
    'nl'      : DutchAnalyzer,
    'da'      : DanishAnalyzer,
    'hi'      : HindiAnalyzer,
    'bn'      : BengaliAnalyzer,
    'zh'      : CJKAnalyzer,  # Chinese
    }

class Retriever(object):
    def __init__(self, k1=None, b=None):
        if k1 is None:
            self.k1 = 1.8  # 0.91
        else:
            self.k1=k1

        if b == None:
            self.b = 0.1  # 0.22
        else:
            self.b=b

    @classmethod
    def get_index_name(self, lang: AnyStr, index_dir: AnyStr = None):
        if index_dir is not None:
            idx_dir = index_dir
        else:
            name = f"{lang}_passage.index"
            idx_dir = os.path.join('data', 'indexes', name)
        logging.info(f"Index dir: {idx_dir}")
        return idx_dir

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
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])

        self.idx_dir = self.get_index_name(lang, index_dir)
        logging.info(f"Creating index in {self.idx_dir}!")

        self.db = db

        # stores index files, poor concurency try NIOFSDirectory instead
        store = SimpleFSDirectory(Paths.get(self.idx_dir))
        # limit max. number of tokens per document.
        # analyzer will not consume more tokens than that
        # analyzer = LimitTokenCountAnalyzer(analyzer, 1048576)
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
        self.doc.add(self.fieldId)

    def createIndex(self):
        with tqdm(total=len(self.db)) as pbar:
            for id, title, passage in self.db:
                self.fieldId.setStringValue(str(id))
                self.fieldTitle.setStringValue(title)
                self.fieldContext.setStringValue(passage)
                self.writer.addDocument(self.doc)
                pbar.update()
        self.commit()

    def commit(self):
        logging.info("Committing index...")
        self.writer.commit()
        self.writer.close()

class Searcher(Retriever):
    def __init__(self, k1=None, b=None):
        super().__init__(k1, b)
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        logging.info(f"Searcher k1: {self.k1}, b: {self.b}")
        self.similarity = BM25Similarity(self.k1, self.b)
        self.searcher = { }
        self.parser_context = { }
        self.parser_title = { }
        self.parser_multi = { }
        self.idx_dir = { }
        self.analyzer = { }
        self.__call__ = self.query

    def addLang(self, lang, index_dir=None):
        """ Initialises index searcher in different languages

        Parameters:
        lang     (str): ['en', 'es', 'de']
        index_path (str): path to index

        Returns:
        """
        idx_dir = self.get_index_name(lang, index_dir=index_dir)
        self.idx_dir[lang] = idx_dir
        if not os.path.exists(idx_dir):
            raise ValueError(f"No index in {idx_dir}!")
        directory = SimpleFSDirectory(Paths.get(idx_dir))
        self.searcher[lang] = IndexSearcher(DirectoryReader.open(directory))
        self.analyzer[lang] = analyzers[lang]()
        self.parser_context[lang] = QueryParser("context", self.analyzer[lang])
        self.parser_title[lang] = QueryParser("title", self.analyzer[lang])
        self.parser_multi[lang] = MultiFieldQueryParser(["title", "context"], self.analyzer[lang])
        self.parser_multi[lang].setDefaultOperator(QueryParser.Operator.OR)
        self.searcher[lang].setSimilarity(self.similarity)

    def get_index_dir(self, lang: str):
        return self.idx_dir[lang]

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
        print("Id:", doc.get('id'))
        # print("Name:", doc.get("title").encode('utf-8'))
        # print("Context:", doc.get("context").encode('utf-8'))
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
        if field not in ['context', 'context_title']:
            raise ValueError(f"Cannot search by {field}!")
        if lang not in self.searcher:
            raise RuntimeError(f"Language '{lang}' not added")

        esccommand = self.parser_context[lang].escape(command)
        if field == 'context':
            query = self.parser_context[lang].parse(esccommand)
        else:
            # query = self.parser_multi[lang].parse(esccommand)
            query = MultiFieldQueryParser.parse(
                esccommand,
                ['title', 'context'],
                [BooleanClause.Occur.SHOULD, BooleanClause.Occur.SHOULD],
                self.analyzer[lang])
        scoreDocs = self.searcher[lang].search(query, topk).scoreDocs

        docs = []
        for scoreDoc in scoreDocs:
            doc = self.getDoc(scoreDoc, lang)
            docs.append(Namespace(score=scoreDoc.score, id=int(doc.get('id')), doc=doc, lang=lang))
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