import tensorflow as tf
tf.get_logger().setLevel(1)
from tensorflow import keras
import numpy as np
import cv2


def preprocessImages(img, show):

    # resize image to 28x28 pixels
    img = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)

    # Adaptive threshold mean
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)

    # bitwise not image
    img = cv2.bitwise_not(img)

    if show:
        cv2.imshow("123",img)
        cv2.waitKey(0)

    img = np.array(img)
    img = img / 255.0

    img = np.expand_dims(img, axis=2)

    return img

def returnNumber(images_to_predict):

    model = keras.models.load_model('my_mnist.model')

    for i in range(len(images_to_predict)):
        images_to_predict[i] = preprocessImages(images_to_predict[i][0], images_to_predict[i][1])

    images_to_predict = np.array(images_to_predict)
    predictions = model.predict(images_to_predict)

    return predictions
