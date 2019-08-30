import numpy as np

from keras.models import Model
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, ZeroPadding1D, Conv1D, Dense, TimeDistributed, concatenate, Flatten
from keras.layers import AveragePooling1D, Add
from keras_contrib.layers import CRF
import keras.backend as K
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        

class NERModel:
    def __init__(self, maxlen, word_dict_size, pos_dict_size, word_vec_size, class_label_count):
        self.maxlen = maxlen
        self.word_dict_size = word_dict_size
        self.word_vec_size = word_vec_size
        self.class_label_count = class_label_count
        self.pos_dict_size = pos_dict_size
        self.model = self._build_model2()
        
    def _build_model(self):
        input_layer = Input(shape=(self.maxlen,), dtype='int32', name='input_layer')
        pos_input_layer = Input(shape=(self.maxlen,), dtype='int32', name='pos_input_layer')
        embedding_layer = Embedding(self.word_dict_size, self.word_vec_size, name='embedding_layer')(input_layer)
        pos_embedding_layer = Embedding(self.pos_dict_size, self.word_vec_size, name='pos_embedding_layer')(pos_input_layer)
        combine_embedding_layer = Add()([embedding_layer, pos_embedding_layer])
        bilstm = Bidirectional(LSTM(32, return_sequences=True))(combine_embedding_layer)
        bilstm_d = Dropout(0.2)(bilstm)
        half_window_size = 2
        paddinglayer = ZeroPadding1D(padding=half_window_size)(embedding_layer)

        conv = Conv1D(nb_filter=32, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
        conv_d = Dropout(0.3)(conv)
        dense_conv = TimeDistributed(Dense(32))(conv_d)
        rnn_cnn_merge = concatenate([bilstm_d, dense_conv], axis=2)
        dense = TimeDistributed(Dense(self.class_label_count))(rnn_cnn_merge)
        dense = Dropout(0.1)(dense) 
        crf = CRF(self.class_label_count, sparse_target=True)
#         crf_output = crf(dense)
        crf_output = crf(dense)
        model = Model(input=[input_layer, pos_input_layer], output=[crf_output])
        model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
        model.summary()
    
        return model
    def _build_model2(self):
        input_layer = Input(shape=(self.maxlen,), dtype='int32', name='input_layer')
        # pos_input_layer = Input(shape=(self.maxlen,), dtype='int32', name='pos_input_layer')
        embedding_layer = Embedding(self.word_dict_size, self.word_vec_size, name='embedding_layer')(input_layer)
        # pos_embedding_layer = Embedding(self.pos_dict_size, self.word_vec_size, name='pos_embedding_layer')(pos_input_layer)
        # combine_embedding_layer = Add()([embedding_layer, pos_embedding_layer])
        bilstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
        bilstm_d = Dropout(0.2)(bilstm)
        half_window_size = 2
        paddinglayer = ZeroPadding1D(padding=half_window_size)(embedding_layer)

        conv = Conv1D(nb_filter=32, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
        conv_d = Dropout(0.3)(conv)
        dense_conv = TimeDistributed(Dense(32))(conv_d)
        rnn_cnn_merge = concatenate([bilstm_d, dense_conv], axis=2)
        dense = TimeDistributed(Dense(self.class_label_count))(rnn_cnn_merge)
        dense = Dropout(0.1)(dense) 
        crf = CRF(self.class_label_count, sparse_target=True)
#         crf_output = crf(dense)
        crf_output = crf(dense)
        model = Model(input=[input_layer], output=[crf_output])
        model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
        model.summary()
    
        return model
    
    def train(self, data, pos_data, label):
        checkpointer = ModelCheckpoint(filepath="../model/bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True) #save_weights_only=True
        history = LossHistory()
        earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='min')
        
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        pos_data = pad_sequences(pos_data, self.maxlen, padding = 'post', truncating='post')
        label = pad_sequences(label, self.maxlen, padding = 'post', truncating='post')
        label = np.expand_dims(label,2)
        self.model.fit([data,pos_data], label,
                       batch_size=512, epochs=30,#validation_data = ([x_test, seq_lens_test], y_test),
                       callbacks=[checkpointer, history,earlystop],
                       verbose=1,
                       validation_split=0.25,
                      )
        
    def predict(self, data, pos_data, id2chunk):
        output_result = []
        self.model.load_weights("../model/bilstm_1102_k205_tf130.w")
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        pos_data = pad_sequences(pos_data, self.maxlen, padding = 'post', truncating='post')
        result = self.model.predict([data,pos_data])
        for i in range(len(result)):
            output_result.append([id2chunk.get(item[1]) for item in np.argwhere(result[i])])
        return output_result

    def train2(self, data, label):
        checkpointer = ModelCheckpoint(filepath="../model/bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True) #save_weights_only=True
        history = LossHistory()
        earlystop = EarlyStopping(monitor='val_loss', patience=3, verbose=2, mode='min')
        
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        label = pad_sequences(label, self.maxlen, padding = 'post', truncating='post')
        label = np.expand_dims(label,2)
        self.model.fit(data, label,
                       batch_size=512, epochs=30,#validation_data = ([x_test, seq_lens_test], y_test),
                       callbacks=[checkpointer, history,earlystop],
                       verbose=1,
                       validation_split=0.25,
                      )
        
    def predict2(self, data, id2chunk):
        output_result = []
        self.model.load_weights("../model/bilstm_1102_k205_tf130.w")
        data = pad_sequences(data, self.maxlen, padding = 'post', truncating='post')
        result = self.model.predict(data)
        for i in range(len(result)):
            output_result.append([id2chunk.get(item[1]) for item in np.argwhere(result[i])])
        return output_result
