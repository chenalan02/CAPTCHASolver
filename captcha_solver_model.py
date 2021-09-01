from numpy.core.fromnumeric import reshape
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization, Reshape
from tensorflow.keras.models import Model

'''
possible_chars = set()

for _, _, files in os.walk('dataset/'):
    for f in files:
        for i in range(5):
            possible_chars.add(f[i])
print(possible_chars)
print(len(possible_chars))
'''

DATADIR = "dataset/"
char_dict = {'2':0,'3':1,'4':2,'5':3,'6':4,'7':5,'8':6,'b':7,'c':8,'d':9,'e':10,'f':11,'g':12,'m':13,'n':14,'p':15,'w':16,'x':17,'y':18}


def process_png(img_path, label, label_mapping):
    
    img = cv2.imread(img_path)
    img = img[:,20:150]
    img = cv2.resize(img, (160, 50))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis = 2)
    img = np.rot90(img, k = 3)

    label = [label[0], label[1], label[2], label[3], label[4]]
    labels = list(map(lambda char: label_mapping[char], label))

    return img, labels

X = []
y = []

for file_name in os.listdir(DATADIR):
    label, file_type = file_name.split('.')
    img_path = os.path.join(DATADIR, file_name)
    if file_type == 'png':
        processed_img, label_encoded = process_png(img_path, label, char_dict)
        X.append(processed_img)
        y.append(label_encoded)

X = np.array(X)
y = np.array(y)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=1)

i = Input(shape = X_train[0].shape)
x = Conv2D(32, (3,3), padding='same', activation='relu')(i)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)

x = Reshape(target_shape=(5, -1))(x)

x = Dense(256, activation="relu")(x)
x = Dense(128, activation="relu")(x)
x = Dense(64, activation="relu")(x)

x = Dense(len(char_dict), activation="softmax")(x)
   
model = Model(i, x)

model.summary()


model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'],)

callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=10, restore_best_weights=True
)

r = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs = 20, callbacks = [callback])

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=2)
num_to_char = {'0':'2','1':'3','2':'4','3':'5','4':'6','5':'7','6':'8','7':'b','8':'c','9':'d','10':'e','11':'f','12':'g','13':'m','14':'n','15':'p','16':'w','17':'x','18':'y'}
nrow = 1
fig=plt.figure(figsize=(20, 5))
for i in range(0,10):
    if i>4:
        nrow = 2
    fig.add_subplot(nrow, 5, i+1)
    plt.imshow(np.rot90(X_val[i], k = 1),cmap='gray')
    plt.title('Prediction : ' + str(list(map(lambda x:num_to_char[str(x)], y_pred[i]))))
    plt.axis('off')
plt.show() 

model.save("captcha_model.h5")
