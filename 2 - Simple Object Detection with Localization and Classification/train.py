
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from model import build_model

""" Global parameters """
global height
global width
global num_classes

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_labels(path):
    df = pd.read_csv(os.path.join(path, "names.csv"), header=None)
    # print(df.head())
    names = df[0].tolist()
    return names

def load_data(path, classes, train=True):
    """ Extracting the images and bounding boxes from csv file. """
    images = []
    bboxes = []
    labels = []

    if train == True:
        df = pd.read_csv(os.path.join(path, "anno_train.csv"), header=None)
    else:
        df = pd.read_csv(os.path.join(path, "anno_test.csv"), header=None)

    for index, row in df.iterrows():
        name = row[0]
        x1 = int(row[1])
        y1 = int(row[2])
        x2 = int(row[3])
        y2 = int(row[4])
        label = int(row[5])

        label_name = classes[label-1]
        if train == True:
            image = os.path.join(path, "car_data", "car_data", "train", label_name, name)
        else:
            image = os.path.join(path, "car_data", "car_data", "test", label_name, name)

        bbox = [x1, y1, x2, y2]

        images.append(image)
        bboxes.append(bbox)
        labels.append(label)

    return images, bboxes, labels

def load_dataset(path, classes, split=0.1):
    train_images, train_bboxes, train_labels = load_data(path, classes, train=True)

    """ Split into training, validation """
    split_size = int(len(train_images) * split)

    train_images, valid_images = train_test_split(train_images, test_size=split_size, random_state=42)
    train_bboxes, valid_bboxes = train_test_split(train_bboxes, test_size=split_size, random_state=42)
    train_labels, valid_labels = train_test_split(train_labels, test_size=split_size, random_state=42)

    test_images, test_bboxes, test_labels = load_data(path, classes, train=False)

    return (train_images, train_bboxes, train_labels), (valid_images, valid_bboxes, valid_labels), (test_images, test_bboxes, test_labels)

def read_image_bbox(path, bbox, label_index):
    """ Image """
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv2.resize(image, (width, height))
    image = (image - 127.5) / 127.5 ## [-1, +1]
    image = image.astype(np.float32)

    """ Bounding box """
    x1, y1, x2, y2 = bbox

    norm_x1 = float(x1/w)
    norm_y1 = float(y1/h)
    norm_x2 = float(x2/w)
    norm_y2 = float(y2/h)
    norm_bbox = np.array([norm_x1, norm_y1, norm_x2, norm_y2], dtype=np.float32)

    label = [0] * num_classes
    label[label_index-1] = 1
    class_label = np.array(label, dtype=np.float32)
    # print(image.shape, norm_bbox.shape, class_label.shape)

    return image, norm_bbox, class_label

def parse(image, bbox, label):
    image, bbox, label = tf.numpy_function(read_image_bbox, [image, bbox, label], [tf.float32, tf.float32, tf.float32])
    image.set_shape((height, width, 3))
    bbox.set_shape((4))
    label.set_shape((num_classes))
    return (image), (bbox, label)

def tf_dataset(images, bboxes, labels, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((images, bboxes, labels))
    ds = ds.map(parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    height = 256
    width = 320
    batch_size = 16
    lr = 1e-4
    num_epochs = 500

    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")
    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Stanford Car Dataset"

    """ Loading labels """
    classes = load_labels(dataset_path)
    num_classes = len(classes)

    """ Dataset """
    (train_images, train_bboxes, train_labels), (valid_images, valid_bboxes, valid_labels), (test_images, test_bboxes, test_labels) = load_dataset(dataset_path, classes, split=0.2)
    print(f"Train: {len(train_images)} - {len(train_bboxes)} - {len(train_labels)}")
    print(f"Valid: {len(valid_images)} - {len(valid_bboxes)} - {len(valid_labels)}")
    print(f"Test : {len(test_images)} - {len(test_bboxes)} - {len(test_labels)}")

    train_ds = tf_dataset(train_images, train_bboxes, train_labels, batch=batch_size)
    valid_ds = tf_dataset(valid_images, valid_bboxes, valid_labels, batch=batch_size)

    # for x, [b, y] in train_ds:
    #     idx = 7
    #     image = x[idx].numpy() * 255.0
    #     x1 = int(b[idx][0] * image.shape[1])
    #     y1 = int(b[idx][1] * image.shape[0])
    #     x2 = int(b[idx][2] * image.shape[1])
    #     y2 = int(b[idx][3] * image.shape[0])
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
    #
    #     text_x = x1
    #     text_y = y1-10
    #     font_size = 1
    #     text = f"{label_names[np.argmax(y[idx])]}"
    #     cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1)
    #
    #     cv2.imwrite("1.png", image)
    #     break

    """ Model """
    model = build_model((height, width, 3))
    model.load_weights(model_path)
    model.compile(
        # loss="binary_crossentropy",
        loss = {
            "bbox": "binary_crossentropy",
            "label": "categorical_crossentropy"
        },
        optimizer=Adam(lr)
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_ds,
        epochs=num_epochs,
        validation_data=valid_ds,
        callbacks=callbacks
    )
