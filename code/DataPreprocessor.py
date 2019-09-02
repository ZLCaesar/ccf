import jieba
from collections import Counter
import json
import pandas as pd
import numpy as np
from pyltp import SentenceSplitter
from PosTagProcessor import POSTagger

import re
class CleanLabel:
    def filter_tags(self,htmlstr):
        htmlstr =str(htmlstr)
        htmlstr = htmlstr.replace('\n','').replace('\t','').replace(' ','')
        htmlstr = htmlstr.replace('\r','')
        #先过滤CDATA
        re_cdata=re.compile('//<!\[CDATA\[[^>]*//\]\]>',re.I) #匹配CDATA
        re_script=re.compile('<\s*script[^>]*>[^<]*<\s*/\s*script\s*>',re.I)#Script
        re_style=re.compile('<\s*style[^>]*>[^<]*<\s*/\s*style\s*>',re.I)#style
        re_br=re.compile('<br\s*?/?>')#处理换行
        re_h=re.compile('</?\w+[^>]*>')#HTML标签
        re_comment=re.compile('<!--[^>]*-->')#HTML注释
        s=re_cdata.sub('',htmlstr)#去掉CDATA
        s=re_script.sub('',s) #去掉SCRIPT
        s=re_style.sub('',s)#去掉style
        s=re_br.sub('\n',s)#将br转换为换行
        s=re_h.sub('',s) #去掉HTML 标签
        s=re_comment.sub('',s)#去掉HTML注释
        #去掉多余的空行
        blank_line=re.compile('\n+')
        s=blank_line.sub('\n',s)
        s=self.replaceCharEntity(s)#替换实体
        return s

    ##替换常用HTML字符实体.
    #使用正常的字符替换HTML中特殊的字符实体.
    #你可以添加新的实体字符到CHAR_ENTITIES中,处理更多HTML字符实体.
    #@param htmlstr HTML字符串.
    def replaceCharEntity(self,htmlstr):

        CHAR_ENTITIES={'nbsp':' ','160':' ',
                    'lt':'<','60':'<',
                    'gt':'>','62':'>',
                    'amp':'&','38':'&',
                    'quot':'"','34':'"',}

        re_charEntity=re.compile(r'&#?(?P<name>\w+);')
        sz=re_charEntity.search(htmlstr)
        while sz:
            entity=sz.group()#entity全称，如&gt;
            key=sz.group('name')#去除&;后entity,如&gt;为gt
            try:
                htmlstr=re_charEntity.sub(CHAR_ENTITIES[key],htmlstr,1)
                sz=re_charEntity.search(htmlstr)
            except KeyError:
                #以空串代替
                htmlstr=re_charEntity.sub('',htmlstr,1)
                sz=re_charEntity.search(htmlstr)
        return htmlstr

class ProcessData:
    def __init__(self, train_path, test_path, maxlen):
        self.maxlen = maxlen
        self.clean = CleanLabel()
        print('数据预处理...')
        self.train_table = pd.read_csv(train_path)
        self.train_table = self.train_table.fillna('')

        self.test_table = pd.read_csv(test_path, dtype=object)
        self.testid = list(self.test_table['id'])
        
        self.test_table = self.test_table.fillna('')
        print(len(set(self.testid)))
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
            # title_list = SentenceSplitter.split(title)
            text = self.clean.filter_tags(text)
            text_list = SentenceSplitter.split(text)

            # for tl in title_list:
            #     out_flag = False
            #     for ent in entities:
            #         if ent in tl:
            #             out_flag = True

            #     if out_flag == False:
            #         continue
            #     if len(tl) > self.maxlen:
            #         sub_tl = self._cut_sent(tl)
            #         for st in sub_tl:
            #             flag = False
            #             for ent in entities:
            #                 if ent in st:
            #                     flag = True
            #             if flag == False:
            #                 continue
                                
            #             temp_label = self._pos_process(st, entities)
            #             label.append(temp_label)
            #             # pos_data.append(self.pos_tagger.get_posseg(st))
            #             ner_data.append(st)
            #     else:
            #         temp_label = self._pos_process(tl, entities)
            #         label.append(temp_label)
            #         # pos_data.append(self.pos_tagger.get_posseg(tl))
            #         ner_data.append(tl)

            append_text = ""

            for tl in text_list:
                out_flag = False
                for ent in entities:
                    if ent in tl:
                        out_flag = True

                if out_flag == False:
                    continue

                if len(append_text + tl) <= self.maxlen:                         #如果当前句子不足100，则继续拼接
                    append_text = append_text + tl

                elif len(tl) > self.maxlen:                                      #如果当前句子直接大于100，则直接拆分
                    if len(append_text) > 0:
                        temp_label = self._pos_process(append_text, entities)
                        label.append(temp_label)
                        ner_data.append(append_text)
                        append_text = ""
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
                else:                                                        #如果当前句子+以拼接的句子长度大于100
                    temp_label = self._pos_process(append_text, entities)
                    label.append(temp_label)
                    ner_data.append(append_text)
                    append_text = tl

                    # temp_label = self._pos_process(tl, entities)
                    # label.append(temp_label)
                    # # pos_data.append(self.pos_tagger.get_posseg(tl))
                    # ner_data.append(tl)
            if len(append_text)>0:
                temp_label = self._pos_process(append_text, entities)
                label.append(temp_label)
                ner_data.append(append_text)
                append_text = tl
        ff = 0
        for _, row in self.test_table.iterrows():
            ids = self.testid[ff]
            ff+=1
            title = row['title']
            text = row['text']
            for w in title:
                vocab[w] = vocab.get(w, 0) + 1
            for w in text:
                vocab[w] = vocab.get(w, 0) + 1

            # title_list = SentenceSplitter.split(title)
            text = self.clean.filter_tags(text)
            text_list = SentenceSplitter.split(text)
            # for tl in title_list:
            #     if len(tl) > self.maxlen:
            #         sub_tl = self._cut_sent(tl)
            #         for st in sub_tl:
            #             test_data.append(st)
            #             # test_pos_data.append(self.pos_tagger.get_posseg(st))
            #             test_id.append(ids)
            #     else:
            #         test_data.append(tl)
            #         # test_pos_data.append(self.pos_tagger.get_posseg(tl))
            #         test_id.append(ids)
            append_text = ""
            for tl in text_list:
                if len(tl) > self.maxlen:
                    if len(append_text) > 0:
                        test_id.append(ids)
                        test_data.append(append_text)
                        append_text = ""
                    sub_tl = self._cut_sent(tl)
                    for st in sub_tl:
                        test_data.append(st)
                        # test_pos_data.append(self.pos_tagger.get_posseg(st))
                        test_id.append(ids)
                elif len(append_text + tl) <= self.maxlen:
                    append_text = append_text + tl
                else:
                    test_id.append(ids)
                    test_data.append(append_text)
                    append_text = tl

                    # test_data.append(tl)
                    # test_id.append(ids)
                    # test_pos_data.append(self.pos_tagger.get_posseg(tl))
            if len(append_text)>0:
                test_id.append(ids)
                test_data.append(append_text)
                append_text = tl

            if test_id[-1] != ids:
                test_id.append(ids)
                test_data.append("")

        char2id = {char: ids+1 for ids, char in enumerate(vocab)}
        char2id ['pad'] = 0
        id2char = {k: v for v, k in char2id.items()}
        print(ff)
        print(len(set(test_id)))
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
        
 
