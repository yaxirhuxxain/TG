from collections import Counter
from itertools import groupby

from treelib import Tree


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


class BeamSearchNode(object):
    def __init__(self, seed, pred, prevNode, logProb, length):
        '''
        :param seed:
        :param pred:
        :param prevNode:
        :param logProb:
        :param length:
        '''
        self.seed = seed
        self.pred = pred
        self.logp = logProb
        self.prevNode = prevNode
        self.len = length

    def eval(self):
        return self.logp / float(self.len - 1 + 1e-6)

    def __eq__(self, other):
        return self.logp == other.logp

    def __ne__(self, other):
        return not (self.logp == other.logp)

    def __lt__(self, other):
        return self.logp < other.logp

    def __gt__(self, other):
        return self.logp > other.logp

    def __le__(self, other):
        return (self.logp < other.logp) or (self.logp == other.logp)

    def __ge__(self, other):
        return (self.logp > other.logp) or (self.logp == other.logp)


def dictOfDict_to_TreePrinter(TREE, seed):
    tree = Tree()
    tree.create_node(seed, seed)
    for key, val in TREE.items():
        for k, v in val.items():
            tree.create_node(v, key + " " + v[0], parent=key)
    tree.show()


def build_vocab(files, vocab_size=30000, tokenizer_fn=None):
    words = Counter()

    for i, name in enumerate(files):
        data = open(name, "r", encoding="utf-8").read()
        for file in data.split('\n'):
            if tokenizer_fn:
                file = " ".join(tokenizer_fn(file))
            words.update(file.split())

    words_to_keep = [i[0] for i in words.most_common(vocab_size - 2)]
    words_to_keep_count_sum = sum(i[1] for i in words.most_common(vocab_size - 2))

    total = sum(words.values())

    # building word to idx encoder
    word_idx = {"<pad>": 0, "<idf>": 1}
    counter = len(word_idx)
    for word in words_to_keep:
        word_idx[word] = counter
        counter += 1

    return word_idx, len(words), 100 * words_to_keep_count_sum / total
