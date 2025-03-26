# src/model.py
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    
    model.add(Flatten())
    
    model.add(Dense(64))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    
    model.add(Dense(24))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(2))
    model.add(Activation('softmax'))
    
    model.summary()
    return model

def compile_and_train(model, X_train, y_train, X_val, y_val, learning_rate, epochs, batch_size):
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_val, y_val), callbacks=[reduce_lr])
    print('Final training loss ', history.history['loss'][-1])
    print('Final training accuracy ', history.history['accuracy'][-1])
    return history

def evaluate_model(model, X_test, y_test):
    testLoss, testAccuracy = model.evaluate(X_test, y_test)
    print('Testing loss ', testLoss)
    print('Testing accuracy ', testAccuracy)
    return testLoss, testAccuracy

def predict(model, X_test):
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    return y_pred, y_pred_class
