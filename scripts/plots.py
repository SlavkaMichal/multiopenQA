import argparse
from matplotlib import pyplot as plt
import numpy as np

# This file is for redoing plots if I need to change anything

parser = argparse.ArgumentParser()
parser.add_argument('--dir', '-d', required=True, type=str)
args = parser.parse_args()

langs = ['ar',
         'da',
         'de',
         'es',
         'fi',
         'fr',
         'hu',
         'it',
         'ja',
         'nl',
         'pl',
         'pt',
         'ru',
         'sv',
         'th',
         'tr',
         'en']


def load_file(dir, lang):
    if lang == 'ht':
        return np.loadtxt(dir + '/hitAtK_ht.csv', delimiter=',')
    else:
        return np.loadtxt(dir + f'/hitAtK_mt_{lang}.csv', delimiter=',')


def plot_ht(langs):
    data = np.loadtxt(f'hitAtK_ht.csv', delimiter=',')
    data[data == -1] = np.inf
    ml = np.cumsum(np.unique(data.min(axis=1), return_counts=True)[1][:-1]) / data.shape[0]
    ys = {}
    for l, v in zip(langs, data.T):
        y = np.cumsum(np.unique(v, return_counts=True)[1][:-1] / len(v))
        ys[l] = y
    colormap = plt.cm.tab20
    x = list(range(1, 21))
    colorst = [colormap(i) for i in np.linspace(0., 1, 17)]
    clv = []

    cm = 1 / 2.54
    page_width = 15.2 * cm
    fig, ax = plt.subplots(figsize=(page_width, 8 * cm))
    ax.axhline(ml[0], linestyle='dashed', label='ml', color='r')
    ax.axhline(ys['en'][16], linestyle='dashed', label='en', color=colorst[0])
    for c, i in zip(colorst, sorted(ys.items(), key=lambda x: x[1][16], reverse=True)):
        clv.append((c, i[0], i[1]))
        ax.plot(x, i[1], label=i[0], color=c)
        ax.scatter([17], i[1][16], marker='x', color=c)
        # if i[0] != 'en':
        #    ax.axhline(i[1][16],c='gray', linewidth=0.5)
    ax.legend(bbox_to_anchor=(1., 1.), loc='upper left')
    ax.set_xticks(x)
    ax.set_yticks(np.linspace(0, 0.6, 13))

    for y in np.linspace(0.1, 0.6, 6):
        ax.axhline(y, c='gray', linewidth=0.5)
    for x in [5, 10, 15]:
        ax.axvline(x, c='gray', linewidth=0.5)

    ax.set_ylabel('Retrieval Accuracy')
    ax.set_xlabel('top k')
    fig.set_size_inches((page_width, 15 * cm))

    labels = ['HT'] + [f'MT {lang}' for lang in langs]
    width = 0.35
    data_mt = {}
    for lang in langs:
        data = np.loadtxt(f'hitAtK_mt_{lang}.csv', delimiter=',')
        data_mt['HT'] = ml[0]
        data_mt[f'{lang}'] = np.unique(data.min(axis=1), return_counts=True)[1][0] / data.shape[0]

    x = np.arange(len(data_mt))
    fig, ax = plt.subplots(figsize=(page_width, 8 * cm))
    data_sorted = {k: v for k, v in sorted(data_mt.items(), key=lambda x: x[1], reverse=True)}
    rects = ax.bar(x, data_sorted.values())
    ax.set_xticks(x)
    ax.set_xticklabels(data_sorted.keys())
    # ax.bar_label(rects, padding=3)
    fig.tight_layout()
    ax.set_ylabel('Retrieval Accuracy')


def plot_correlation():
    # language , number of articles, nubmer of passages
    wiki_data = {
        'da': (266599, 745507, 'o'),  # germanic
        'de': (2573726, 10870498, 'o'),
        'en': (6294702, 21015324, 'o'),
        'sv': (3196626, 3612145, 'o'),
        'nl': (2054338, 3154149, 'o'),

        'fi': (508881, 1225855, 's'),  # finnish
        'hu': (487415, 1542793, 's'),

        'es': (1681092, 7809593, '^'),  # spanish
        'fr': (2326064, 9559330, '^'),
        'it': (1691467, 6072981, '^'),
        'pt': (1066054, 3523032, '^'),

        'pl': (1472246, 3283254, 'D'),  # slavic
        'ru': (1721794, 6898313, 'D'),

        'ar': (1115094, 1894884, 'P'),  # semitic
        'th': (139618, 151351, 'P'),  # kra-dal
        'tr': (401588, 845427, 'P'),  # turkic
        'ja': (1267288, 7141013, 'P'),  # japonic
        }

    data = np.loadtxt(f'hitAtK_ht.csv', delimiter=',')
    data[data == -1] = np.inf
    # ml = np.cumsum(np.unique(data.min(axis=1),return_counts=True)[1][:-1])/data.shape[0]
    ys = {}
    for l, v in zip(langs, data.T):
        y = np.cumsum(np.unique(v, return_counts=True)[1][:-1] / len(v))
        ys[l] = (wiki_data[l][0], wiki_data[l][2], y[16])
    colormap = plt.cm.tab20
    colorst = [colormap(i) for i in np.linspace(0., 1, 17)]
    clv = []

    cm = 1 / 2.54
    page_width = 15.2 * cm
    fig, ax = plt.subplots(figsize=(page_width, 8 * cm))
    # ax.scatter(ml[0], linestyle='dashed', label='multilingual', color='r')
    x = [w[0] for w in wiki_data.values()]
    for c, i in zip(colorst, sorted(ys.items(), key=lambda x: x[1], reverse=True)):
        clv.append((c, i[0], i[1]))
        print(i[1])
        ax.scatter(i[1][0], i[1][2], marker=i[1][1], color=c, label=i[0])
        # if i[0] != 'en':
        #    ax.axhline(i[1][16],c='gray', linewidth=0.5)
    # ax.legend(loc=4)
    ax.legend(bbox_to_anchor=(1., 1.), loc='upper left')

    # ax.set_xticks(x)
    ax.set_yticks(np.linspace(0.05, 0.6, 12))

    for y in np.linspace(0.1, 0.6, 6):
        ax.axhline(y, c='gray', linewidth=0.5)
    for x in np.array([1, 2, 3, 4, 5, 6]) * 1e6:
        ax.axvline(x, c='gray', linewidth=0.5)

    ax.set_ylabel('Retrieval Accuracy')
    ax.set_xlabel('Number of Wikipedia articles')
    fig.set_size_inches((page_width, 15 * cm))


def wiki_size():
    wiki_data = {'da': (266599, 745507, 'o'),
                 'de': (2573726, 10870498, 'o'),
                 'en': (6294702, 21015324, 'o'),
                 'sv': (3196626, 3612145, 'o'),
                 'nl': (2054338, 3154149, 'o'),
                 'fi': (508881, 1225855, 's'),
                 'hu': (487415, 1542793, 's'),
                 'es': (1681092, 7809593, '^'),
                 'fr': (2326064, 9559330, '^'),
                 'it': (1691467, 6072981, '^'),
                 'pt': (1066054, 3523032, '^'),
                 'pl': (1472246, 3283254, 'D'),
                 'ru': (1721794, 6898313, 'D'),
                 'ar': (1115094, 1894884, 'P'),
                 'th': (139618, 151351, 'P'),
                 'tr': (401588, 845427, 'P'),
                 'ja': (1267288, 7141013, 'P')}

    wiki_data = {i[0]: i[1] for i in sorted(wiki_data.items(), key=lambda x: x[1][0], reverse=True)}

    labels = [k for k in wiki_data.keys()]
    norm_art = np.array([v[0] for v in wiki_data.values()]) // 1000_000
    norm_psgs = np.array([v[1] for v in wiki_data.values()]) // 1000_000
    # norm_art = norm_art/ norm_art.sum()
    # norm_psgs = norm_psgs/norm_psgs.sum()

    fig, ax = plt.subplots()
    x = np.arange(len(wiki_data))
    width = 0.35

    rects1 = ax.bar(x - width / 2, norm_art, width, label='Articles')
    rects2 = ax.bar(x + width / 2, norm_psgs, width, label='Passages')
    ax.set_ylabel('Size')
    ax.set_xlabel('Wikipedias')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    fig.tight_layout()


def plot_all():
    data = np.array([[0.14993016, 0.2154276, 0.19230172, 0.18221325, 0.09622846],
                     [0.14046252, 0.14899891, 0.15481217, 0.18221325, 0.09622846],
                     [0.11407729, 0.11330126, 0.09839504, 0.1050477, 0.07512029]])

    width = 0.22
    labels = ['HT only', 'MT question', 'MT only']
    fig, ax = plt.subplots()
    x = list(range(3))
    rects_small = ax.bar(np.array(x) - 1.5 * width, data[:, 1], width, label='Pivot')
    rects_tag = ax.bar(np.array(x) - 0.5 * width, data[:, 2], width, label='Tag')
    rects_oqmr = ax.bar(np.array(x) + 0.5 * width, data[:, 3], width, label='OQMR')
    rects_oqer = ax.bar(np.array(x) + 1.5 * width, data[:, 4], width, label='OQER')

    ax.set_ylabel('EM Score')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
