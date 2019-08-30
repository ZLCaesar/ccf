import jieba.posseg as psg
class POSTagger:
    def __init__(self):
        
        self.pos2id, self.id2pos = self._get_pos_dict()
        
    def get_posseg(self, sent):
        pos_arr = []
        for item in psg.lcut(sent):
            item = list(item)
            word = item[0]
            tag = item[1]
            for i in range(len(word)):
                pos_arr.append(self.pos2id.get(tag))
            
        return pos_arr
    
    def _get_pos_dict(self):
        pos2id = {'el':0, 'v': 1, 'n': 2, 'x': 3, 'r': 4, 'uj': 5, 'y': 6, 'm': 7, 'd': 8, 't': 9, 'p': 10, 'mq': 11, 'a': 12, 'zg': 13, 'k': 14, 'l': 15, 'ad': 16, 'vn': 17,
         'c': 18, 'eng': 19, 'q': 20, 'nr': 21, 'f': 22, 'nz': 23, 'vd': 24, 'ns': 25, 'j': 26, 'nt': 27, 'u': 28, 'b': 29, 'ul': 30, 'ud': 31, 'i': 32, 'nrt': 33, 'vg': 34,
         's': 35, 'ng': 36, 'tg': 37, 'ag': 38, 'an': 39, 'z': 40, 'ug': 41, 'yg': 42, 'g': 43, 'df': 44, 'uv': 45, 'e': 46, 'nrfg': 47, 'uz': 48, 'h': 49, 'rz': 50, 'o': 51,
         'rr': 52, 'vq': 53, 'vi': 54, 'dg': 55, 'mg': 56}
        id2pos = {v: k for k, v in pos2id.items()}
        
        return pos2id, id2pos
    
    
