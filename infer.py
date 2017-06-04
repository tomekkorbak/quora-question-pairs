
from datetime import datetime
import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout, Input, LSTM, Embedding, Bidirectional, Lambda
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ProgbarLogger, TensorBoard
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras_tqdm import TQDMNotebookCallback

from utils import load_embeddings, extract_questions_from_dataframe, save_submission


class Config(object):
    VOCABULARY_SIZE = 1193514
    EMBEDDING_DIMENSION = 200
    OFFSET = 3
    OOV_TOKEN = 0  # out of vocabulary
    EOS_TOKEN = 1  # end of sentence
    PAD_TOKEN = 2  # padding to max sentence length
    MAX_SENTENCE_LENGTH = 60
    DENSE_LAYER_SIZE = 150
    DROPOUT = 0.4

    def stamp(self, comment):
        return '{date:%Y%m%d_%H%M}_{comment}'.format(
            date=datetime.now(), comment=comment)


# In[4]:

train_dataframe = pd.read_csv('train.csv')
current_config = Config()

embedding_weights, word2idx = load_embeddings(
    'glove.twitter.27B.200d.txt',
    config=current_config
)

questions_A, questions_B, labels = extract_questions_from_dataframe(
    train_dataframe,
    config=current_config,
    word2idx=word2idx,
    prediction_mode=False
)

# In[4]:

questions_A.shape


# In[13]:


def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(
        K.maximum(
            K.sum(K.square(x - y), axis=1, keepdims=True),
            K.epsilon()
        )
    )


shared_lstm_layer = LSTM(
    units=200,
    return_sequences=False,
    dropout=0.4,
    recurrent_dropout=0.3
)

shared_embedding_layer = Embedding(
    input_dim=current_config.VOCABULARY_SIZE + current_config.OFFSET,
    output_dim=current_config.EMBEDDING_DIMENSION,
    input_length=current_config.MAX_SENTENCE_LENGTH,
    weights=[embedding_weights],
    trainable=False
)

input_A = Input(shape=(current_config.MAX_SENTENCE_LENGTH,))
embeddings_A = shared_embedding_layer(input_A)
sentence_representation_A = shared_lstm_layer(embeddings_A)
normalized_A = BatchNormalization()(sentence_representation_A)

input_B = Input(shape=(current_config.MAX_SENTENCE_LENGTH,))
embeddings_B = shared_embedding_layer(input_B)
sentence_representation_B = shared_lstm_layer(embeddings_B)
normalized_B = BatchNormalization()(sentence_representation_B)

distance = Lambda(euclidean_distance)(
    [normalized_A, normalized_B]
)

predictions = Dense(1, activation='sigmoid')(distance)

model = Model(inputs=[input_A, input_B], outputs=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()

model.load_weights('20170604_1251_1.h5')
test_dataframe = pd.read_csv('test.csv')
test_questions_A, test_questions_B, _ = extract_questions_from_dataframe(
    test_dataframe,
    config=current_config,
    word2idx=word2idx,
    prediction_mode=True
)
predictions = model.predict(
    x=[test_questions_A, test_questions_B],
    batch_size=8192,
    verbose=1
)
save_submission(predictions, current_config)