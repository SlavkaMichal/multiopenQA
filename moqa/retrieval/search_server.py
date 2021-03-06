from ..common.utils import get_root
from .connection import Server, Client
from argparse import Namespace

config = Namespace(
    host = 'knot5.fit.vutbr.cz',
    port = 8765,
    index_dir = '/home/michal/data/indexes/wiki_dpr_en.index',
    b=0.22,
    k1=0.95,
    topk=5,
    search_fields='context', # 'context' or 'context_title'
    lucene=False,
    langs=['en']
)

def main(config: Namespace):
    if config.server:
        server = Server(config.port)
        server.run()
    else:
        client = Client(
                host=config.host,
                port=config.port,
                index_dir=config.index_dir,
                b=config.b,
                k1=config.k1,
                topk=config.topk,
                search_fields=config.search_fields,
                ret_lucene=config.lucene,
                langs=config.langs)
        queries = [dict(question='Where was Van Gogh born?', lang='en', id='0'),
                   dict(question='Which is the most widely spoken language in the world', lang='en', id='1')]
        result = client.search(queries)
        print(result)
