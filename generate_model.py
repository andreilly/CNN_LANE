import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# Import necessary items from Keras
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers


def build_CNN(input_shape, pooling_size):
    # Create the actual neural network here
    neural_net = Sequential()
    # Normalizes incoming inputs. First layer needs the input shape to work
    neural_net.add(BatchNormalization(input_shape=input_shape))

    # Below layers were re-named for easier reading of model summary; this not necessary
    # Conv Layer 1
    neural_net.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv1'))

    # Conv Layer 2
    neural_net.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv2'))

    # Pooling 1
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Conv Layer 3
    neural_net.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv3'))
    neural_net.add(Dropout(0.2))

    # Conv Layer 4
    neural_net.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv4'))
    neural_net.add(Dropout(0.2))

    # Conv Layer 5
    neural_net.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv5'))
    neural_net.add(Dropout(0.2))

    # Pooling 2
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Conv Layer 6
    neural_net.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv6'))
    neural_net.add(Dropout(0.2))

    # Conv Layer 7
    neural_net.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Conv7'))
    neural_net.add(Dropout(0.2))

    # Pooling 3
    neural_net.add(MaxPooling2D(pool_size=pooling_size))

    # Upsample 1
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconv 1
    neural_net.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv1'))
    neural_net.add(Dropout(0.2))

    # Deconv 2
    neural_net.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv2'))
    neural_net.add(Dropout(0.2))

    # Upsample 2
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconv 3
    neural_net.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv3'))
    neural_net.add(Dropout(0.2))

    # Deconv 4
    neural_net.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv4'))
    neural_net.add(Dropout(0.2))

    # Deconv 5
    neural_net.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv5'))
    neural_net.add(Dropout(0.2))

    # Upsample 3
    neural_net.add(UpSampling2D(size=pooling_size))

    # Deconv 6
    neural_net.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Deconv6'))

    # Final layer - only including one channel so 1 filter
    neural_net.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation = 'relu', name = 'Final'))

    return neural_net


def main():
    # Load training images
    train_images = pickle.load(open("full_CNN_train.p", "rb" ))

    # Load image labels
    labels = pickle.load(open("full_CNN_labels.p", "rb" ))

    # Make into arrays as the neural network wants these
    train_images = np.array(train_images)
    labels = np.array(labels)

    # Normalize labels - training images get normalized to start in the network
    labels = labels / 255

    # Shuffle images along with their labels, then split into training/validation sets
    train_images, labels = shuffle(train_images, labels)
    # Test size may be 10% or 20%
    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)

    # Batch size, epochs and pool size below are all paramaters to fiddle with for optimization
    batch_size = 128
    n_epochs = 10
    pooling_size = (2, 2)
    input_shape = X_train.shape[1:]

    # Create the neural network
    neural_net = build_CNN(input_shape, pooling_size)

    # Using a generator to help the model use less data
    # Channel shifts help with shadows slightly
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)

    # Compiling and training the model
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')
    neural_net.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
    epochs=n_epochs, verbose=1, validation_data=(X_val, y_val))

    # Freeze layers since training is done
    neural_net.trainable = False
    neural_net.compile(optimizer='Adam', loss='mean_squared_error')

    # Save model architecture and weights
    neural_net.save('full_CNN_model.h5')

    # Show summary of model
    neural_net.summary()

if __name__ == '__main__':
    main()
