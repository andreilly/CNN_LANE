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


def build_CNN(input_shape, pooling_size):
    
    # Initialize the convolutional neural network
    neural_net = Sequential()
    
    # Batch normalization to normalize the input
    neural_net.add(BatchNormalization(input_shape=input_shape))

    # Convolution Layer 1
    neural_net.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Convolution Layer 2
    neural_net.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling Layer 1
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Convolution Layer 3 with dropout
    neural_net.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    neural_net.add(Dropout(0.2))

    # Convolution Layer 4 with dropout
    neural_net.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    neural_net.add(Dropout(0.2))

    # Convolution Layer 5 with dropout
    neural_net.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    neural_net.add(Dropout(0.2))

    # Pooling Layer 2
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Convolution Layer 6 with dropout
    neural_net.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    neural_net.add(Dropout(0.2))

    # Convolution Layer 7 with dropout
    neural_net.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    neural_net.add(Dropout(0.2))

    # Pooling Layer 3
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Upsampling Layer 1
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconvolution Layer 1 with dropout
    neural_net.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))

    # Deconvolution Layer 2 without dropout
    neural_net.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))

    # Upsampling Layer 2
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconvolution Layer 3 without dropout
    neural_net.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))

    # Deconvolution Layer 4 without dropout
    neural_net.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))

    # Deconvolution Layer 5 without dropout
    neural_net.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))

    # Upsampling Layer 3
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconvolution Layer 6
    neural_net.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Output Layer with a single filter for grayscale output
    neural_net.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    return neural_net


def main():
    # Retrieve training and labels images
    train_images = pickle.load(open("Dataset/CNN_train.p", "rb" ))
    labels = pickle.load(open("Dataset/CNN_labels.p", "rb" ))

    # Convert to arrays for neural network
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize the labels; training images normalization is included within the network
    labels = labels / 255

    # Mix and split the dataset into a training and validation set
    train_images, labels = shuffle(train_images, labels)
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    # Set hyperparameters for the training process
    n_epochs = 10
    pooling_size = (2, 2)
    batch_size = 128
    input_shape = X_train.shape[1:]

    # Construct the neural network
    neural_net = build_CNN(input_shape, pooling_size)

    # Data augmentation generator
    augmentation = ImageDataGenerator(channel_shift_range=0.2)
    augmentation.fit(X_train)

    # Compile and train the network
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')
    neural_net.fit_generator(augmentation.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=n_epochs, verbose=1, validation_data=(X_val, y_val))

    # Lock the network's layers for inference
    neural_net.trainable = False
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')

    # Save the model's structure and weights
    neural_net.save('full_CNN_model.h5')

    # Output a summary of the model's architecture
    neural_net.summary()

if __name__ == '__main__':
    main()
