import os
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import medial_axis
from skimage import img_as_ubyte
from skimage.morphology import skeletonize, thin

dir_path = os.path.dirname(os.path.realpath(__file__))


class OCR:

    def __init__(self, test_dir = dir_path + '/train_images/'):
        self.learning_rate = 0.001
        self.test_dir = test_dir
        self.data = []

    def format_y(self, y):

        res = [0]*128

        if y == None:
            return res

        for i in range(len(y)):

            if y[i] == None:
                continue

            res[y[i]-2304] = 1

        return res

    def get_character(self, text):

        chars = []
        split = text.split('_')
        split = split[3:]

        if len(split) <= 0:
            print ('Image label not found for ', text)
            return

        split[-1] = split[-1][:-4] #remove .png from last split

        for c in range(len(split)):
            chars.append(int(split[c]))

        return chars

    def load_data(self):

        if not self.test_dir:
            print ("Test directory could not be found. Please specify the correct path.")
            return

        images_name = [f for f in listdir(self.test_dir) if isfile(join(self.test_dir, f))]
        X = []
        y = []

        for file_name in images_name:
            file_path = self.test_dir
            img = cv2.imread(file_path + file_name, 0)

            img = self.preprocess(img)
            X.append(img.reshape(64,64,1))

            chars = self.get_character(file_name) 
            y.append(self.format_y(chars))

        return np.array(X), np.array(y)

    def preprocess(self, img):
        cv2.normalize(img, img, 0, 255, cv2.NORM_MINMAX)
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        ret, ret_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) 
        resz_img = cv2.resize(ret_img, (64,64))

        kernel1 = np.ones((3,3), np.uint8)
        erosion = cv2.erode(resz_img, kernel1, iterations = 1)

        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2)
        ret, res_img = cv2.threshold(resz_img, 0, 255, cv2.THRESH_BINARY_INV)
        res_img = cv2.filter2D(res_img,-1,kernel)
        return res_img

    def make_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3,3), strides=(1, 1),
                 activation='relu', input_shape=(64,64,1)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(128, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(256, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(512, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='sigmoid'))

        self.model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=self.learning_rate), metrics=[categorical_accuracy])

    def train(self):
        X,Y = self.load_data() #loads train data, preprocess it and return in desired format

        X = X.astype('float32')

        train_datagen = ImageDataGenerator(
            rotation_range=20,
            rescale=1./255,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest')

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size=0.5)
        X_test=X_test/255.
        train_generator = train_datagen.flow(X_train, Y_train, batch_size=32)
        validation_generator = train_datagen.flow(X_validate, Y_validate, batch_size=32)

        if isfile(join(dir_path, '/my_model_new.h5')):
            self.model = load_model('my_model_new.h5')
        else:
            self.make_model()

        self.model.fit_generator(
                    train_generator,
                    steps_per_epoch=len(X_train)/32,
                    epochs=50,
                    validation_data=validation_generator,
                    validation_steps=len(X_validate)/32)

        score,acc = self.model.evaluate(X_test, Y_test)
        self.model.save('my_model_new.h5')
        print ('score: ', score, 'accuracy: ', acc)




ocr = OCR()
ocr.model = load_model('model.h5')

def predict(imagePath, model = ocr):
    img = cv2.imread(imagePath, 0)
    img1 = model.preprocess(img)
    img1 = img1.astype('float32')
    img1 = img1/255.0
    img1 = img1.reshape((1,64,64,1))
    preds = model.model.predict(img1)
    preds[preds >=0.5] = 1
    preds[preds < 0.5] = 0

    res_c = []
    for pred in range(len(preds[0])):
        if preds[0][pred] == 0:
            continue

        res = pred + 2304
        res_c.append(res)
    ret = map(chr, res_c)
    #print ("Characters predicted in image are:")
    #print (res_c)
    #print (", ".join(ret))


    return res_c
