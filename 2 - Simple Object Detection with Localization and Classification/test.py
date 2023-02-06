
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from tqdm import tqdm
import tensorflow as tf
from sklearn.metrics import accuracy_score
from train import load_dataset, create_dir, load_labels

""" Global parameters """
global height
global width
global num_classes
global label_names

def cal_iou(y_true, y_pred):
    x1 = max(y_true[0], y_pred[0])
    y1 = max(y_true[1], y_pred[1])
    x2 = min(y_true[2], y_pred[2])
    y2 = min(y_true[3], y_pred[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    true_area = (y_true[2] - y_true[0] + 1) * (y_true[3] - y_true[1] + 1)
    bbox_area = (y_pred[2] - y_pred[0] + 1) * (y_pred[3] - y_pred[1] + 1)

    iou = intersection_area / float(true_area + bbox_area - intersection_area)
    return iou

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Path """
    dataset_path = "/media/nikhil/Seagate Backup Plus Drive/ML_DATASET/Stanford Car Dataset"
    model_path = os.path.join("files", "model.h5")

    """ Load the model """
    model = tf.keras.models.load_model(model_path)

    """ Hyperparameters """
    height = 256
    width = 320

    classes = load_labels(dataset_path)
    num_classes = len(classes)

    """ Dataset """
    _, _, (test_images, test_bboxes, test_labels) = load_dataset(dataset_path, classes, split=0.2)
    test_images = test_images[:1000]
    test_bboxes = test_bboxes[:1000]
    test_labels = test_labels[:1000]
    print(f"Test : {len(test_images)} - {len(test_bboxes)} - {len(test_labels)}")

    """ Prediction """
    mean_iou = []
    pred_labels = []

    for image, true_bbox, true_labels in tqdm(zip(test_images, test_bboxes, test_labels), total=len(test_images)):
        """ Extracting the name """
        name = image.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        x = cv2.resize(image, (width, height))
        x = (x - 127.5) / 127.5
        x = np.expand_dims(x, axis=0)

        """ Bounding box """
        true_x1, true_y1, true_x2, true_y2 = true_bbox

        """ Prediction """
        pred_bbox, label = model.predict(x, verbose=0)
        pred_bbox = pred_bbox[0]
        label_index = np.argmax(label[0])
        pred_labels.append(label_index+1)

        """ Rescale the bbox points. """
        pred_x1 = int(pred_bbox[0] * image.shape[1])
        pred_y1 = int(pred_bbox[1] * image.shape[0])
        pred_x2 = int(pred_bbox[2] * image.shape[1])
        pred_y2 = int(pred_bbox[3] * image.shape[0])

        """ Calculate IoU """
        iou = cal_iou(true_bbox, [pred_x1, pred_y1, pred_x2, pred_y2])
        mean_iou.append(iou)

        """ Plot bbox on image """
        image = cv2.rectangle(image, (true_x1, true_y1), (true_x2, true_y2), (255, 0, 0), 2) ## BLUE
        image = cv2.rectangle(image, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 0, 255), 2) ## RED

        """ Plot predicted class label and score """
        font_size = 1
        pred_class_name = classes[label_index]
        text = f"{pred_class_name}"
        cv2.putText(image, text, (pred_x1, pred_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 2)

        """ Plot true class label """
        font_size = 1
        true_class_name = classes[true_labels-1]
        text = f"{true_class_name}"
        cv2.putText(image, text, (true_x1, true_y1-10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)

        """ Plot IoU on image """
        x = 50
        y = 50
        font_size = 1
        cv2.putText(image, f"IoU: {iou:.4f}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 2)
        # print(f"Image Name: {name} - IoU: {iou:.4f} - Pred label: {pred_class_name} - True label: {true_class_name}")

        """ Saving the image """
        cv2.imwrite(f"results/{name}", image)

    """ Mean IoU """
    score = np.mean(mean_iou, axis=0)
    mean_acc = accuracy_score(test_labels, pred_labels)
    print(f"Mean IoU: {score:.4f} - Acc: {mean_acc:.4f}")
