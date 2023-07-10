import cv2
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global Variables
mouse_mode = False
pt = (0, 0)
color = (200, 200, 200)
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


model.fit(train_data, train_label, epochs=10)
train_result = model.evaluate(test_data, test_label)
print('loss :', train_result[0])
print('Acc  :', train_result[1])

predict_result = model.predict(test_data)
predict_label = np.argmax(predict_result, axis=1)


# Selection
wrong_result = []

for i in range(0, len(test_label)):
    if(predict_label[i] != test_label[i]):
        wrong_result.append(i)

print("Error : " + str(len(wrong_result)))

sample = random.choices(population=wrong_result, k=16)

# Plot
plt.figure(figsize=(14, 12))

for i, id in enumerate(sample):
    plt.subplot(4, 4, i+1)
    plt.imshow(test_data[id].reshape(28, 28), cmap='gray')
    plt.title("Label : " + str(test_label[id]) + " | Predict : " + str(predict_label[id]))
    plt.axis('off')

plt.show()

# Save
# plt.savefig("fig1.png", dpi=1500)

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
        Test = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
        Test = cv2.cvtColor(Test, cv2.COLOR_BGR2GRAY)
        Test = Test / 255.0
        Test = Test.reshape(1, 784).astype('float32')
        predict_result = model.predict(Test)
        prediction = np.argmax(predict_result, axis=1)
        print(prediction)
        image = np.full((280, 280, 3), 0, np.uint8)
        Predict_Window = np.full((280, 500, 3), 255, np.uint8)
        cv2.putText(Predict_Window, 'Predict : ' + str(prediction[0]), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
        cv2.imshow("Predict", Predict_Window)


cv2.imshow("PaintCV", image)
cv2.putText(Predict_Window, 'Predict : ', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2)
cv2.imshow("Predict", Predict_Window)
cv2.setMouseCallback("PaintCV", onMouse)

while True:

    cv2.imshow("PaintCV", image)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
