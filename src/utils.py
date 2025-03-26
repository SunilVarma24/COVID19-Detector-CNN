# src/utils.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def visualize_random_images(images, labels, num_images=5):
    r = np.random.choice(len(images), num_images, replace=False)
    for i in r:
        plt.imshow(images[i], cmap='gray')
        plt.title(f'label:{labels[i]}')
        plt.show()

def print_classification_report(y_true, y_pred_class):
    print(classification_report(y_true, y_pred_class))

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
