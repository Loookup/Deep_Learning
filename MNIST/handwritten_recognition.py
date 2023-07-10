import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
import copy
import math
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Variables
mouse_mode = False
pt = (0, 0)
color = (255, 255, 255)
thickness = 9
image = np.full((280, 280, 3), 0, np.uint8)
Test = np.full((280, 280, 3), 0, np.uint8)
Predict_Window = np.full((280, 500, 3), 255, np.uint8)
prediction = None

# MNIST Data Set import
mnist = tf.keras.datasets.mnist
(train_data, train_label), (test_data, test_label) = mnist.load_data()

# Normalize
train_data, test_data = train_data/255.0, test_data/255.0

# Flatten
train_data = train_data.reshape(60000, 784).astype('float32')
test_data = test_data.reshape(10000, 784).astype('float32')


# Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Model Compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
              
callbacks = [tf.keras.callbacks.TensorBoard('./logs_keras')]

model.fit(train_data, train_label, epochs=10, verbose=1, callbacks=callbacks)

train_result = model.evaluate(test_data, test_label)

print('loss :', train_result[0])
print('Acc  :', train_result[1])
print("----Model Construction Complete----")


def onMouse(event, x, y, flags, param):

    global pt, mouse_mode, color, thickness, image, Test, prediction, Predict_Window

    if event == cv2.EVENT_LBUTTONDOWN:

        pt = (x, y)

        mouse_mode = True

    elif event == cv2.EVENT_MOUSEMOVE:

        if mouse_mode == True:

            cv2.line(image, pt, (x, y), color, thickness)

            pt = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:

        mouse_mode = False

        cv2.line(image, pt, (x, y), color, thickness)

    elif event == cv2.EVENT_RBUTTONDOWN:

        image, prediction = Find_and_Sort(image)

        Predict_Window = np.full((280, 500, 3), 255, np.uint8)

        result = 0

        for i in range(0, len(prediction)):

            sig = 1

            for sq in range(0, i):

                sig = sig * 10

            result += sig * prediction[len(prediction) - i - 1]
            
        cv2.putText(Predict_Window, 'Predict : ' + str(result), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

        cv2.imshow("Predict", Predict_Window)

def Find_and_Sort(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    th_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)[1]
    
    # For Eliminate Noise
    # kernel = np.ones((5, 17), np.uint8)
    # morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours = cv2.findContours(th_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = [cv2.minAreaRect(c) for c in contours[1]]

    w_list = []

    for center, size, angle in rects:

        w, h = center

        w_list.append(int(w))

    for idx in range(len(w_list)-1):

        criterion = np.argmin(w_list[idx:len(w_list):1])

        criterion = criterion + idx

        if idx != criterion:
            
            Duplicated_list = copy.deepcopy(w_list)

            w_list[idx] = Duplicated_list[criterion]

            w_list[criterion] = Duplicated_list[idx]


    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle) for center, size, angle in rects]

    dup_candidates = copy.deepcopy(candidates)

    for idx in range(len(w_list)):

        if candidates[idx][0][0] != w_list[idx]:

            for seq in range(len(w_list)):

                if dup_candidates[seq][0][0] == w_list[idx]:

                    candidates[idx] = dup_candidates[seq]

                    break

                else:
                    continue

    image , prediction_list = Model_Fit(image, candidates)

    cv2.imshow("PaintCV", image)

    return image, prediction_list


def Model_Fit(image, candidates):

    idx = 0

    prediction_list = []

    for candidate in candidates:

        crop = rotate_number(image, candidate)

        crop = fill(crop)

        Test = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_LINEAR)

        Test = cv2.cvtColor(Test, cv2.COLOR_BGR2GRAY)

        Test = Test / 255.0

        Test = Test.reshape(1, 784).astype('float64')

        predict_result = model.predict(Test)

        prediction = np.argmax(predict_result, axis=1)

        prediction_list.append(prediction[0])

        cv2.imshow(str(idx + 1) + "th : result = " + str(prediction[0]), crop)

        pts = np.int32(cv2.boxPoints(candidate))

        cv2.polylines(image, [pts], True, (0, 255, 255), 2)
        
        idx += 1

    return image, prediction_list


def fill(image):

    h, w = image.shape[:2]

    h, w = int(h), int(w)

    if w % 2 == 1:
        w += 1
    if h % 2 == 1:
        h += 1

    fill = np.full((h + 20, h + 20, 3), 0, np.uint8)

    for y in range(-int(h/2), int(h/2)-1):

        for x in range(-int(w/2), int(w/2)-1):

            if x + int(h/2) + 10 < h + 20 and y + int(h/2) + 10 < h + 20 and x + int(w/2) < w and y + int(h/2) < h:

                try :

                    fill[y + int(h/2) + 10, x + int(h/2) + 10] = image[y + int(h/2), x + int(w/2)]

                except Exception as ex:

                    print(h, w, x + int(h/2) + 10, y + int(h/2) + 10, x + int(w/2), y + int(h/2))
                    print(ex)
            else:

                continue

    return fill


def rotate_number(image, rect):

    center, (w, h), angle = rect

    if w > h :
        w, h = h, w
        angle += 90

    size = image.shape[1::-1]

    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)

    rot_img = cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)

    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    return crop_img


cv2.imshow("PaintCV", image)

cv2.putText(Predict_Window, 'Predict : ', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)

cv2.imshow("Predict", Predict_Window)

cv2.setMouseCallback("PaintCV", onMouse)

while True:

    cv2.imshow("PaintCV", image)

    if cv2.waitKey(1) == 27:

        cv2.destroyAllWindows()

        print("----End Handwriting Recognition with Multi-Digits----")

        break
