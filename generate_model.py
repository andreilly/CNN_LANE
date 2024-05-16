import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


def build_CNN(input_dim, pooling_size):
    # Initialize the CNN (Convolutional Neural Network)
    cnn_model = Sequential()

    # Input normalization layer
    cnn_model.add(BatchNormalization(input_shape=input_dim))

    #region Convolutional Layers (with pooling and dropout) 
    # Conv Layer 1
    cnn_model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    cnn_model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    cnn_model.add(MaxPooling2D(pool_size=pooling_size))

    # Conv Layer 3
    cnn_model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    cnn_model.add(Dropout(0.2))

    # Conv Layer 4
    cnn_model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    cnn_model.add(Dropout(0.2))

    # Conv Layer 5
    cnn_model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    cnn_model.add(Dropout(0.2))

    # Pooling 2
    cnn_model.add(MaxPooling2D(pool_size=pooling_size))

    # Conv Layer 6
    cnn_model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    cnn_model.add(Dropout(0.2))

    # Conv Layer 7
    cnn_model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    cnn_model.add(Dropout(0.2))

    # Pooling 3
    cnn_model.add(MaxPooling2D(pool_size=pooling_size))

    # Upsample 1
    cnn_model.add(UpSampling2D(size=pooling_size))
    #endregion

    
    #region Deconvolutional Layers 
    # Deconv 1
    cnn_model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))

    # Deconv 2
    cnn_model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))

    # Upsample 2
    cnn_model.add(UpSampling2D(size=pooling_size))

    # Deconv 3
    cnn_model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))

    # Deconv 4
    cnn_model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))

    # Deconv 5
    cnn_model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))

    # Upsample 3
    cnn_model.add(UpSampling2D(size=pooling_size))

    # Deconv 6
    cnn_model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    cnn_model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))
    #endregion

    return cnn_model


def main():
    # Set model parameters
    pooling_size = (2, 2) 
    b_size = 128
    n_epochs = 10

    # Load the training data and labels
    train_dataset = pickle.load(open("Dataset/CNN_train.p", "rb" ))
    labels_dataset = pickle.load(open("Dataset/CNN_labels.p", "rb" ))

    # Convert datasets to numpy arrays for Keras
    train_dataset = np.array(train_dataset)
    labels_dataset = np.array(labels_dataset)

    # Normalize the labels dataset to match the image normalization
    labels_dataset = labels_dataset / 255

    # Shuffle and split the dataset into training and validation sets
    train_dataset, labels_dataset = shuffle(train_dataset, labels_dataset)
    X_train, X_val, y_train, y_val = train_test_split(train_dataset, labels_dataset, test_size=0.2)
    input_dim = X_train.shape[1:]

    # Data augmentation for improved training
    augmentor = ImageDataGenerator(channel_shift_range=0.2)
    augmentor.fit(X_train)

    # Build and compile the CNN
    neural_net = build_CNN(input_dim, pooling_size)

    # Train the CNN
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')
    neural_net.fit_generator(augmentor.flow(X_train, y_train, batch_size=b_size), steps_per_epoch=len(X_train)/b_size, epochs=n_epochs, verbose=1, validation_data=(X_val, y_val))

    # Finalize the model
    neural_net.trainable = False
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')

    # Save the CNN architecture and weights
    neural_net.save('Models/CNN_model.h5')

    # Display a summary of the CNN
    neural_net.summary()

if __name__ == '__main__':
    main()
