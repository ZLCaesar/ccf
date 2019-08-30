import jieba
from collections import Counter
import json
import pandas as pd
import numpy as np
from pyltp import SentenceSplitter
from PosTagProcessor import POSTagger

class ProcessData:
    def __init__(self, train_path, test_path, maxlen):
        self.maxlen = maxlen
        print('数据预处理...')
        self.train_table = pd.read_csv(train_path)
        self.train_table = self.train_table.fillna('')

        self.test_table = pd.read_csv(test_path)
        self.test_table = self.test_table.fillna('')
        # self.pos_tagger = POSTagger()
        self.char2id, self.id2char, self.test_id, self.test_data, self.test_pos_data, self.ner_data, self.pos_data, self.label = self._process_data()
        self.chunk2id = {'O':0, 'B':1, 'I':2}
        self.id2chunk = {0:'O', 1:'B', 2:'I'}

    def _find_word_position(self, sent, word):
        if word not in sent:
            return []
        ret_arr = []
        begin = 0
        while(word in sent):
            index = sent.index(word)
            ret_arr.append((begin + index, begin + index + len(word)))
            sent = sent[index + len(word):]
            begin = begin + index + len(word)
        return ret_arr

    def _pos_process(self, text, ents):
        temp_label = ['O'] * len(text)
        for ent in ents:
            temp_pos = self._find_word_position(text, ent)
            for pair in temp_pos:
                temp_label[pair[0]] = 'B'
                temp_label[pair[0]+1: pair[1]] = 'I' * (pair[1] - pair[0]-1)
                
        return temp_label

    def _cut_sent(self, sent):
        arr = []
        num = len(sent)//self.maxlen

        for i in range(num):
            arr.append(sent[i*self.maxlen:(i+1)*self.maxlen])
            
        arr.append(sent[(i+1)*self.maxlen:])
        return arr

    def _process_data(self):
        vocab = {}
        test_data = []
        test_pos_data = []
        test_id = []
        ner_data = []
        pos_data = []
        label = []
        for _, row in self.train_table.iterrows():
            _id = row['id']
            title = row['title']
            text = row['text']
            unknownEntities = row['unknownEntities']
            for w in title:
                vocab[w] = vocab.get(w, 0) + 1

            for w in text:
                vocab[w] = vocab.get(w, 0) + 1

            if unknownEntities == '':
                continue

            entities = unknownEntities.split(';')
            title_list = SentenceSplitter.split(title)
            text_list = SentenceSplitter.split(text)

            for tl in title_list:
                out_flag = False
                for ent in entities:
                    if ent in tl:
                        out_flag = True

                if out_flag == False:
                    continue
                if len(tl) > self.maxlen:
                    sub_tl = self._cut_sent(tl)
                    for st in sub_tl:
                        flag = False
                        for ent in entities:
                            if ent in st:
                                flag = True
                        if flag == False:
                            continue
                                
                        temp_label = self._pos_process(st, entities)
                        label.append(temp_label)
                        # pos_data.append(self.pos_tagger.get_posseg(st))
                        ner_data.append(st)
                else:
                    temp_label = self._pos_process(tl, entities)
                    label.append(temp_label)
                    # pos_data.append(self.pos_tagger.get_posseg(tl))
                    ner_data.append(tl)

            
            for tl in text_list:
                out_flag = False
                for ent in entities:
                    if ent in tl:
                        out_flag = True

                if out_flag == False:
                    continue
                if len(tl) > self.maxlen:
                    sub_tl = self._cut_sent(tl)
                    for st in sub_tl:
                        flag = False
                        for ent in entities:
                            if ent in st:
                                flag = True
                        if flag == False:
                            continue

                        temp_label = self._pos_process(st, entities)
                        label.append(temp_label)
                        # pos_data.append(self.pos_tagger.get_posseg(st))
                        ner_data.append(st)
                else:
                    temp_label = self._pos_process(tl, entities)
                    label.append(temp_label)
                    # pos_data.append(self.pos_tagger.get_posseg(tl))
                    ner_data.append(tl)
                
        for _, row in self.test_table.iterrows():
            _id = row['id']
            title = row['title']
            text = row['text']
            for w in title:
                vocab[w] = vocab.get(w, 0) + 1
            for w in text:
                vocab[w] = vocab.get(w, 0) + 1

            title_list = SentenceSplitter.split(title)
            text_list = SentenceSplitter.split(text)
            for tl in title_list:
                if len(tl) > self.maxlen:
                    sub_tl = self._cut_sent(tl)
                    for st in sub_tl:
                        test_data.append(st)
                        # test_pos_data.append(self.pos_tagger.get_posseg(st))
                        test_id.append(_id)
                else:
                    test_data.append(tl)
                    # test_pos_data.append(self.pos_tagger.get_posseg(tl))
                    test_id.append(_id)
            for tl in text_list:
                if len(tl) > self.maxlen:
                    sub_tl = self._cut_sent(tl)
                    for st in sub_tl:
                        test_data.append(st)
                        # test_pos_data.append(self.pos_tagger.get_posseg(st))
                        test_id.append(_id)
                else:
                    test_data.append(tl)
                    test_id.append(_id)
                    # test_pos_data.append(self.pos_tagger.get_posseg(tl))

        char2id = {char: ids+1 for ids, char in enumerate(vocab)}
        char2id ['pad'] = 0
        id2char = {k: v for v, k in char2id.items()}

        return char2id, id2char, test_id, test_data, test_pos_data, ner_data, pos_data, label
        

    def get_train_data(self):
        train_data = []
        train_lable = []

        for i in range(len(self.ner_data)):
            data = self.ner_data[i]
            label = self.label[i]

            train_data.append(np.asarray([self.char2id.get(w) for w in data]))
            train_lable.append(np.asarray([self.chunk2id.get(w) for w in label]))


        return train_data, self.pos_data, train_lable
        

    def get_dict(self):
        return self.chunk2id, self.id2chunk, self.char2id, self.id2char

                

    def get_test_data(self, idnize=True, combine=True):
        test_data = []
        
        for i in range(len(self.test_data)):
            text = self.test_data[i]
            test_data.append(np.asarray([self.char2id.get(w) for w in text]))
                
        return test_data, self.test_pos_data
        
 
