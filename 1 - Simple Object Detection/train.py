
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
H = 512
W = 512

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.1):
    """ Extracting the images and bounding boxes from csv file. """
    images = []
    bboxes = []

    df = pd.read_csv(os.path.join(path, "bbox.csv"))
    for index, row in df.iterrows():
        name = row["name"]
        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        image = os.path.join(path, "images", name)
        bbox = [x1, y1, x2, y2]

        images.append(image)
        bboxes.append(bbox)

    """ Split into training, validation and testing. """
    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(bboxes, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image_bbox(path, bbox):
    """ Image """
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w, _ = image.shape
    image = cv2.resize(image, (W, H))
    image = (image - 127.5) / 127.5 ## [-1, +1]
    image = image.astype(np.float32)

    """ Bounding box """
    x1, y1, x2, y2 = bbox

    norm_x1 = float(x1/w)
    norm_y1 = float(y1/h)
    norm_x2 = float(x2/w)
    norm_y2 = float(y2/h)
    norm_bbox = np.array([norm_x1, norm_y1, norm_x2, norm_y2], dtype=np.float32)

    return image, norm_bbox

def parse(x, y):
    x, y = tf.numpy_function(read_image_bbox, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([4])
    return x, y

def tf_dataset(images, bboxes, batch=8):
    ds = tf.data.Dataset.from_tensor_slices((images, bboxes))
    ds = ds.map(parse).batch(batch).prefetch(10)
    return ds

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 16
    lr = 1e-4
    num_epochs = 500
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Human-Detection"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    train_ds = tf_dataset(train_x, train_y, batch=batch_size)
    valid_ds = tf_dataset(valid_x, valid_y, batch=batch_size)

    # for x, y in valid_ds:
    #     idx = 7
    #     image = x[idx].numpy() * 255.0
    #     x1 = int(y[idx][0] * image.shape[1])
    #     y1 = int(y[idx][1] * image.shape[0])
    #     x2 = int(y[idx][2] * image.shape[1])
    #     y2 = int(y[idx][3] * image.shape[0])
    #     image = cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    #     cv2.imwrite("1.png", image)
    #     break

    """ Model """
    model = build_model((H, W, 3))
    model.load_weights(model_path)
    model.compile(
        loss="binary_crossentropy",
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
