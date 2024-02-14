import os, logging
from collections import Counter
from multiprocessing import Pool
from functools import partial
from . import util

logger = logging.getLogger(__name__)

class Vocabulary(object):
    fake = {'[PAD]': 0, '[SOS]': 1, '[EOS]': 2, '[UNK]': 3, '[SEP]': 4, '[MASK]': 5}
    num = len(fake)
    for i in range(0, 10):
        fake['[unused{}]'.format(i+1)] = num+i

    def __init__(self, name, vocab_path, min_cnt=5, tokenizer=None):
        self.name = name
        self.vocab_path = vocab_path
        self.size = None
        self.vocab = None
        self.tokenizer = tokenizer
        self.min_cnt = min_cnt

        if self.vocab_path is not None and os.path.isfile(self.vocab_path):
            self.restore()

    @property
    def word2id(self):
        return self.vocab['word2id']

    @property
    def id2word(self):
        return self.vocab['id2word']

    @property
    def word2cnt(self):
        return self.vocab['word2cnt']

    def restore(self):
        logger.info('load vocab from saved file: %s', self.vocab_path)
        self.vocab = util.load_json(self.vocab_path)
        self.size = len(self.vocab['word2id'])

    def _build_vocab(self, cnt):
        word2id = self.fake.copy()
        id2word = [k for k, v in sorted(word2id.items(), key=lambda x:x[1])]
        for word in sorted(cnt):
            if cnt[word] >= self.min_cnt:
                word2id[word] = len(word2id)
                id2word.append(word)
        self.size = len(word2id)
        self.vocab = {'word2id': word2id, 'id2word': id2word, 'word2cnt': dict(cnt)}


    @staticmethod
    def count(sents, tokenizer=None):
        cnt = Counter()
        if tokenizer:
            for sent in sents:
                for word in tokenizer(sent):
                    cnt.update([word])
        else:
            for sent in sents:
                for word in sent:
                    cnt.update([word])
        return cnt

    def update_chinese_bert_vocab(self, vocab):
        word2id = dict()
        word2id['[UNK]'] = vocab['[UNK]']
        unk_words = dict()
        for k, v in self.vocab['word2id'].items():
            if k in vocab:
                word2id[k] = vocab[k]
            else:
                word2id[k] = vocab['[UNK]']
                if k not in self.fake:
                    unk_words[k] = self.vocab['word2cnt'][k]
        if '[CLS]' in vocab:
            word2id['[SOS]'] = vocab['[CLS]']
            word2id['[CLS]'] = vocab['[CLS]']
        if '[SEP]' in vocab:
            word2id['[EOS]'] = vocab['[SEP]']
            word2id['[SEP]'] = vocab['[SEP]']
        if '[MASK]' in vocab:
            word2id['[MASK]'] = vocab['[MASK]']
        unk_words = [x[0] for x in sorted(unk_words.items(), key=lambda x: x[1], reverse=True)]
        logging.info('num of unk words: %s, first 20:%s', len(unk_words), unk_words[0:20])
        #for i, word in enumerate(unk_words[0:99]):
        #    word2id[word] = vocab['[unused{}]'.format(i+1)]

        id2word = {v:k for k, v in word2id.items()}
        self.vocab['word2id'] = word2id
        self.vocab['id2word'] = id2word
        self.size = len(vocab)

    def build_vocab(self, sents, save=True, parallel=0):
        if self.vocab is None:
            logger.info('building vocab')
            if parallel >0:
                num = len(sents)
                batch_size = (num//parallel +1)
                batchs = [sents[i:i+batch_size] for i in range(0, len(sents), batch_size)]
                pool = Pool(parallel)
                cnts = pool.map(partial(self.count, tokenizer=self.tokenizer), batchs)
                cnt = sum(cnts, Counter())
            else:
                cnt = self.count(sents, tokenizer=self.tokenizer)
            self._build_vocab(cnt)
            if save:
                util.dump_json(self.vocab, self.vocab_path)
                logger.info('vocab saved to :%s', self.vocab_path)
        logger.info('vocab size:%s', self.size)

    def seq(self, sents, max_len=None, sos=None, eos=None, add_sos=True, add_eos=True, w2i=None, pad=None):
        if w2i is None:
            w2i = self.vocab['word2id']
        if pad is None:
            pad = w2i['[PAD]']
        add_len = 0
        if add_sos:
            add_len += 1
            sos = sos or w2i['[SOS]']
        if add_eos:
            add_len += 1
            eos = eos or w2i['[EOS]']
        seqs = []
        seq_lens = []
        for sent in sents:
            if self.tokenizer:
                tokens = self.tokenizer(sent)
            else:
                tokens = sent
            inds = [w2i[w] if w in w2i else w2i['[UNK]'] for w in tokens]
            if max_len is not None:
                inds = inds[0:max_len-add_len]
            if add_sos:
                inds = [sos] + inds
            if add_eos:
                inds += [eos]
            seq_lens.append(len(inds))
            if max_len is not None:
                inds += [pad] * (max_len - len(inds))
            seqs.append(inds)
        return seqs, seq_lens

    def deseq(self, seqs):
        id2w = self.vocab['id2word']
        sents = []
        for seq in seqs:
            sent = []
            for word_id in seq:
                sent.append(id2w[word_id])
            sents.append(sent)
        return sents



