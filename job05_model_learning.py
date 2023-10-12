import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import *
from tensorflow.keras.layers import *

X_train, X_test, Y_train, Y_test = np.load(
    './crawling_data/news_data_max_21_wordsize_12454.npy', allow_pickle=True
)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
# Embedding(wordsize, 차원 축소(12461 차원을 300으로 줄인다), input_length=max)
# 자연어 처리할 때 사용, 하나의 단어마다 좌표값을 가지게 함, 단어 하나마다 의미를 주기 위해서
# 차원 축소를 하는 이유 : 차원이 많아질수록 좌표들의 거리가 멀어지는 것(차원의 저주)을 방지하기 위해서
# 의미 공간상의 벡터화
model.add(Embedding(12461, 300, input_length=21))
# 문장은 1줄 -> 1차원
model.add((Conv1D(32, kernel_size=5, padding='same', activation='relu')))
# 아무일도 하지 않음, 빼도 상관 없음, 하지만 Conv1D와 항상 같이 사용(값이 잘 못 나왔을 때 값을 바꿔줄 수도 있기 때문)
model.add(MaxPooling1D(pool_size=1))
# 뒤에 LSTM이 또 나오기 때문에 return_sequences = True
model.add(LSTM(128, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.3))
# 마지막 LSTM이기에 return_sequences 값을 주지 않음
model.add(LSTM(64, activation='tanh'))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(6, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))
model.save('./models/news_category_classification_model_{}.h5'.format(fit_hist.history['val_accuracy'][-1]))
plt.plot(fit_hist.history['val_accuracy'], label='validation accuracy')
plt.plot(fit_hist.history['accuracy'], label='accuracy')
plt.legend()
plt.show()

