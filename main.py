# main.py
from src import data, model, utils, config
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    # 1. Preprocessing images
    images, labels = data.load_images_labels(config.DATA_PATH, img_size=config.IMG_SIZE)
    
    # Encoding labels
    labels_onehot, label_encoder = data.encode_labels(labels)
    
    # 2. Visualizing the images with labels
    utils.visualize_random_images(images, labels, num_images=5)
    
    # 3. Train-test-split
    X_train, X_test, y_train, y_test = train_test_split(images, labels_onehot, test_size=0.15, random_state=0)
    print("X_train data shape:", X_train.shape)
    print("X_test data shape:", X_test.shape)
    print("y_train data shape:", y_train.shape)
    print("y_test data shape:", y_test.shape)
    
    # 4. Train-val-split
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=0)
    print("X_train data shape:", X_train.shape)
    print("X_val data shape:", X_val.shape)
    print("y_train data shape:", y_train.shape)
    print("y_val data shape:", y_val.shape)
    
    # 5. Image Data Generation
    from keras.preprocessing.image import ImageDataGenerator
    datagen = ImageDataGenerator(
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    train_datagen = datagen.flow(X_train, y_train, batch_size=44)
    
    # 6. CNN model
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
    cnn_model = model.build_cnn_model(input_shape)
    
    # 7. Model training
    history = model.compile_and_train(cnn_model, X_train, y_train, X_val, y_val,
                                      learning_rate=config.LEARNING_RATE,
                                      epochs=config.EPOCHS,
                                      batch_size=config.BATCH_SIZE)
    
    # 8. Model evaluation
    testLoss, testAccuracy = model.evaluate_model(cnn_model, X_test, y_test)
    
    # 9. Classification metrics
    y_pred, y_pred_class = model.predict(cnn_model, X_test)
    y_true = np.argmax(y_test, axis=1)
    utils.print_classification_report(y_true, y_pred_class)
    
    # 10. Confusion matrix
    utils.plot_confusion_matrix(y_true, y_pred_class, class_names=['Class 0', 'Class 1'])

if __name__ == '__main__':
    main()