import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12, 8)

import gc
import os
import ast
import sys
import configparser
import glob
import librosa
import librosa.display
from scipy.stats import mode

from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K


# 경준
def load_songs(song_folder):
    song_specs = []
    idx_to_genre = []
    genre_to_idx = {}
    genres = []
    for genre in os.listdir(song_folder):
        genre_to_idx[genre] = len(genre_to_idx)
        idx_to_genre.append(genre)
        genre_folder = os.path.join(song_folder, genre)
        for song in os.listdir(genre_folder):
            if song.endswith('.au'):
                signal, sr = librosa.load(os.path.join(genre_folder, song))
                melspec = librosa.feature.melspectrogram(signal, sr=sr).T[:1280,]
                song_specs.append(melspec)
                genres.append(genre_to_idx[genre])
    return song_specs, genres, genre_to_idx, idx_to_genre

song_specs, genres, genre_to_idx, idx_to_genre = load_songs('/home/douwe/genres')
song_specs[0].shape



librosa.display.specshow(librosa.power_to_db(song_specs[101].T,
                                              ref=np.max),
                          y_axis='mel', fmax=8000,
                          x_axis='time')



# 준영
def show_spectogram(show_genre):
    show_genre = genre_to_idx[show_genre]
    specs = []
    for spec, genre in zip(song_specs, genres):
        if show_genre == genre:
            specs.append(spec)
            if len(specs) == 25:
                break
    if not specs:
        return 'not found!'
    x = np.concatenate(specs, axis=1)
    x = (x - x.min()) / (x.max() - x.min())
    plt.imshow((x *20).clip(0, 1.0))

show_spectogram('classical')

# 희유
def lstm_model(input_shape):
    '''Create LSTM MODEL

    Parameters
    ----------
    input_shape : tuple
        (input_shape_row, input_shape_col)

    Returns
    -------
    model : object
        keras LSTM object

    Note
    ----
    Input
        케라스모델에게 Input shape를 알려주며 텐서로 인스턴스화.
    LSTM
        units : output의 공간을 할당합니다. 현재는 128개의 공간을 만듬.
        return_sequences : bool, 출력 시퀀스의 마지막 출력을 반환할지 아니면 전체 시퀀스를 반환할지 지정.
    Dense
        입력과 출력을 모두 연결. fully connected
        입력 뉴런 수에 상관없이 출력 뉴런 수를 자유롭게 설정할 수 있기 때문에 출력층으로 많이 사용.
        activation 함수를 지정.
    Model
        Input 과 output을 지정하여 모델을 생성합니다.
        현재 구성 : inputs -> lstm_1 -> dense2
    optimizers
        최적화 함수를 설정. 현재는 SGD. 이외에는 RMSprop, Adagrad, Adadelta, Adam 등 존재
        lr : Learning rate. lr > 0
        momentum : 최적화 방향의 가속도를 지정하고 진동을 줄임.
        decay : 업데이트에 대한 학습 속도 설정
        nesterov : Nesterov momentum 적용 유무

        모멘텀 : 누적된 과거 그래디언트가 지향하고 있는 어떤 방향을 현재 그래디언트에 보정하려는 방식.
        네스테로프 모멘텀 : 모멘텀 알고리즘을 개선한 것으로 관성의 효과로 최적값을 지나칠 수 있는 문제를 방지.
    compile
        Cost Function과 최적화 함수를 설정하고 모델을 compile합니다.
    '''

    inputs = Input(shape=input_shape, name='input')
    lstm_1 = LSTM(units=128, return_sequences=False)(inputs)
    dense2 = Dense(10, activation='softmax')(lstm_1)
    model = Model(inputs=[inputs], outputs=[dense2])
    sgd = keras.optimizers.SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model

#준혁
def cnn_model(input_shape):
    inputs = Input(input_shape)
    x = inputs
    levels = 64

    for level in range(3):
        x = Conv1D(levels, 3, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        levels *= 2

    # Global Layers
    x = GlobalMaxPooling1D()(x)

    for fc in range(2):
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)

    labels = Dense(10, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[labels])
    sgd = keras.optimizers.SGD(lr=0.0003, momentum=0.9, decay=1e-5, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    return model


model = cnn_model((128, 128))
model.summary()

# 완기
def split_10(x, y):
    s = x.shape
    s = (s[0] * 10, s[1] // 10, s[2])
    return x.reshape(s), np.repeat(y, 10, axis=0)


genres_one_hot = keras.utils.to_categorical(genres, num_classes=len(genre_to_idx))

x_train, x_test, y_train, y_test = train_test_split(
    np.array(song_specs), np.array(genres_one_hot),
    test_size=0.1, stratify=genres)

x_test, y_test = split_10(x_test, y_test)
x_train, y_train = split_10(x_train, y_train)

x_train.shape, y_train.shape

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
      min_delta=0,
      patience=3,
      verbose=0,
      mode='auto')

    # Fit the model
history = model.fit(x_train, y_train,
      batch_size=128,
      epochs=100,
      verbose=1,
      validation_data=(x_test, y_test),
      callbacks = [earlystop])


model.save('zoo/15/song_classify.h5')

def unsplit(values):
    chunks = np.split(values, 100)
    return np.array([np.argmax(chunk) % 10 for chunk in chunks])

pred_values = model.predict(x_test)
predictions = unsplit(pred_values)
truth = unsplit(y_test)
accuracy_score(predictions, truth)