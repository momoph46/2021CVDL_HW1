from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.image as img
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import numpy as np


cifar_label=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','trunk']


(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
x_train = X_train.astype('float32') / 255
x_test = X_test.astype('float32') / 255

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

parameter_loss='categorical_crossentropy'
parameter_optimizer='adam'
parameter_batchsize=128
parameter_epochs=100
parameter_validationsplit=0.1

def construct_vgg():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    # model.add(Dropout(0.25))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(128, activation='relu'))  # 1024
    model.add(Dense(128, activation='relu'))  # 1024
    model.add(Dropout(0.25))
    model.add(Dense(10, activation='softmax'))

    return model

def show_acc_loss():
    acc = img.imread('acc.png')
    loss = img.imread('loss.png')
    plt.imshow(acc)
    plt.show()
    plt.imshow(loss)
    plt.show()

def show_hyperparameter():
    print('Loss:　', parameter_loss)
    print('Optimizer: ', parameter_optimizer)
    print('Batch size: ', parameter_batchsize)
    print('Epochs:　', parameter_epochs)
    print('Validation split: ', parameter_validationsplit)

def show_model():
    model = construct_vgg()
    print(model.summary())

def show_trainimage():
    fig = plt.gcf()
    fig.set_size_inches(35, 35)
    for i in range(0, 9):
        ax = plt.subplot(3, 3, 1 + i)
        # ax.
        ax.imshow(x_train[i])
        ax.set_title(cifar_label[Y_train[i][0]])
    plt.show()

def test_image(n):
    cifar_model=load_model('cifar10_model.h5')
    predict = cifar_model.predict(x_test[n].reshape(-1, 32, 32, 3))

    testlebal = cifar_label[Y_test[n][0]]
    plt.title(testlebal)
    plt.imshow(x_test[n])
    plt.show()

    fig = plt.figure()
    x = np.arange(len(cifar_label))
    plt.bar(x, predict[0])
    plt.xticks(x, cifar_label)
    plt.xlabel('cifar_label')
    plt.ylabel('Acc')
    plt.title('')
    fig.set_figheight(6)
    fig.set_figwidth(10)
    plt.show()

def train():
    model=construct_vgg()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x=x_train,
                        y=y_train,
                        batch_size=128,
                        epochs=50,
                        validation_split=0.1,
                        )
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('test loss: ', test_loss)
    print('test acc: ', test_acc)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    model.save('cifar10_model.h5')
