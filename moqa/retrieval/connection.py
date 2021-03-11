import pickle
import socket
import lucene
from ..common.utils import get_root
from .retrieval import Searcher
import time
import os
from moqa.common import config
import logging

logging.basicConfig(
    format=f"%(asctime)s:%(filename)s:%(lineno)d:%(levelname)s: %(message)s",
    filename=config.log_file,
    level=config.log_level)


class Connection(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.conn = self.sock
        self.open = False
        self.client = True

    def __del__(self):
        self.stop()

    def recvall(self):
        data = self.conn.recv(4096)
        n = pickle.loads(data)
        data = data[len(pickle.dumps(n)):]

        while n != len(data):
            data += self.conn.recv(4096)

        data = pickle.loads(data)
        return data

    def sendall(self, msg: dict) -> None:
        data = pickle.dumps(msg)
        n = len(data)
        data = pickle.dumps(n) + data
        self.conn.sendall(data)
        return

    def accept(self):
        (conn, addr) = self.sock.accept()
        logging.info(f"Connected to {addr}")
        self.conn = conn

    def connect(self, host, port):
        self.sock.connect((host, port))
        self.open = True
        self.client = True

    def close(self, stop=False):
        logging.info("Stopping")
        if not self.open:
            return
        msg = {'stop': stop}
        self.sendall(msg)
        try:
            self.conn.close()
        except OSError:
            pass

    def stop(self):
        """ Stopping server"""
        if not self.client and self.open:
            self.conn.close()
        try:
            self.sock.close()
        except OSError:
            pass


class Server(Connection):
    def __init__(self, port):
        logging.info("Initializing Connection")
        super().__init__()
        logging.info("Initializing VM")
        lucene.initVM(vmargs=['-Djava.awt.headless=true'])
        self.sock.bind((socket.gethostname(), port))
        self.sock.listen()

        self.write_intensity = 1000
        self.index_dir = os.path.join(get_root(), 'data', 'indexes')
        self.dataset = 'wiki'
        self.b = 0.22
        self.k1 = 0.92
        self.topk = 5
        self.field = 'context'
        self.ret_lucene = False
        self.langs = ['en']

    def run(self):
        stop = False
        while not stop:
            self.accept()
            stop = self.search()
        self.stop()

    def search(self):
        tally = 0
        while True:
            start_recv = time.time()
            recv = self.recvall()
            stop_recv = time.time()
            if 'init' in recv:
                logging.info("Initialising searcher")
                if 'write_intensity' in recv['init']:
                    self.write_intensity = recv['init']['write_intensity']
                if 'index_dir' in recv['init']:
                    self.index_dir = recv['init']['index_dir']
                else:
                    self.sendall(dict(error='index_dir required for initialisation!'))
                if 'b' in recv['init']:
                    self.b = recv['init']['b']
                if 'k1' in recv['init']:
                    self.k1 = recv['init']['k1']
                if 'topk' in recv['init']:
                    self.topk = recv['init']['topk']
                if 'field' in recv['init']:
                    self.field = recv['init']['field']
                    if not self.field == 'context' or not self.field == 'context_title':
                        self.sendall(dict(error=f"Cannot search by {self.field}!"))
                        raise ValueError(f"Cannot search by {self.field}!")
                if 'ret_lucene' in recv['init']:
                    self.ret_lucene = recv['init']['ret_lucene']

                if 'langs' in recv['init']:
                    self.langs = recv['init']['langs']

                self.searcher = Searcher(k1=self.k1, b=self.b)
                for lang in self.langs:
                    self.searcher.addLang(lang, self.index_dir)
                self.sendall(dict(ok=True))

                logging.info(f"Write intensity: {self.write_intensity}")
                logging.info(f"Index path: {self.index_dir}")
                logging.info(f"b:          {self.b}")
                logging.info(f"k1:         {self.k1}")
                logging.info(f"topk:       {self.topk}")
                logging.info(f"field:      {self.field}")
                logging.info(f"ret_lucene:   {self.ret_lucene}")
                logging.info(f"langs:      {self.langs}")

            if 'search' in recv:
                response = dict(result=[])
                for query in recv['search']:
                    question = query['question']
                    lang = query['lang']
                    search_id = query['id']
                    try:
                        docs = self.searcher.query(question, lang, self.topk, field=self.field)
                    except Exception as e:
                        if 'error' in response:
                            response['error'].append(str(e))
                        else:
                            response['error'] = [str(e)]
                        continue
                    documents = []
                    for doc in docs:
                        document = { 'score': doc.score, 'id': doc.doc.getField['id'], 'lang': doc.lang }
                        if self.ret_lucene:
                            document['lucene'] = doc.doc
                        documents.append(document)
                    search_result = dict(id=search_id, docs=documents)
                    response['result'].append(search_result)
                self.sendall(response)

            if 'stop' in recv:
                logging.info("Stopping")
                if recv['stop']:
                    self.close()
                    return True
                else:
                    self.close()
                break

            end_request = time.time()
            if tally % self.write_intensity == 0 and self.write_intensity != 0:
                logging.info("Request number: ", tally)
                logging.info("Recv took:   ", stop_recv - start_recv)
                logging.info("Request took: ", end_request - stop_recv)
                logging.info("Request type: ", recv.keys())
            tally += 1
        return False


class Client(Connection):
    def __init__(self,
                 host: str,
                 port: int,
                 index_dir: str,  # data/indexes
                 write_intensity: int = 0,
                 b: float = None,  # 0.22
                 k1: float = None,  # 0.92
                 topk: int = None,
                 search_fields: str = None,  # context, context_title
                 ret_lucene: bool = False,  # score, luceneDoc, id
                 langs: list = None  # en
                 ):
        logging.info("Initializing Client")
        super().__init__()
        self.sock.connect((host, port))
        msg = { 'init': dict(write_intensity=write_intensity, index_dir=index_dir, b=b, k1=k1,
                             topk=topk, field=search_fields, ret_lucene=ret_lucene, langs=langs) }
        # remove uninitialised values
        # uninitialised values will be initialised as server default values
        msg['init'] = { k: v for k, v in msg['init'].items() if v is not None }
        for k, v in msg['init'].items():
            logging.info("{0: <12}: {1}".format(k, v))
        self.sendall(msg)
        response = self.recvall()
        if 'error' in response:
            raise RuntimeError("Initialisation failed: ", response['error'])
        if 'ok' not in response:
            raise RuntimeError("Did not received confirmation: ", response)

    def search(self, queries: list) -> list:
        """
        Returns list of search results
        :param queries: List of queries [{'question': question, 'lang': language, 'id': search id}]
        :return:    [ {'id': search id, docs:[{'score': score, 'id': id, 'lang': language, 'lucene': lucene document}]]
        """
        query_list = []
        for i, query in enumerate(queries):
            q = dict(question=query['question'], lang=query['lang'], id=i if 'id' not in query else query['id'])
            if 'id' in query:
                q['id'] = query['id']
            query_list.append(q)
        msg = { 'search': query_list }
        self.sendall(msg)
        result = self.recvall()
        if 'error' in result:
            raise RuntimeError("Search failed: ", result['error'])
        if 'result' not in result:
            raise RuntimeError("Result not included: ", result)
        return result['result']
