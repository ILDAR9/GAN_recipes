import glob

import numpy as np
from keras import Sequential, Input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import ReLU
from keras.layers import Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing import image
from scipy.stats import entropy


def build_generator():
    gen_model = Sequential()

    gen_model.add(Dense(input_dim=100, units=2048))
    gen_model.add(ReLU())

    gen_model.add(Dense(256 * 8 * 8))
    gen_model.add(BatchNormalization())
    gen_model.add(ReLU())
    gen_model.add(Reshape((8, 8, 256), input_shape=(256 * 8 * 8,)))
    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(128, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(64, (5, 5), padding='same'))
    gen_model.add(ReLU())

    gen_model.add(UpSampling2D(size=(2, 2)))

    gen_model.add(Conv2D(3, (5, 5), padding='same'))
    gen_model.add(Activation('tanh'))
    return gen_model


def build_discriminator():
    dis_model = Sequential()
    dis_model.add(
        Conv2D(128, (5, 5),
               padding='same',
               input_shape=(64, 64, 3))
    )
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(256, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Conv2D(512, (3, 3)))
    dis_model.add(LeakyReLU(alpha=0.2))
    dis_model.add(MaxPooling2D(pool_size=(2, 2)))

    dis_model.add(Flatten())
    dis_model.add(Dense(1024))
    dis_model.add(LeakyReLU(alpha=0.2))

    dis_model.add(Dense(1))
    dis_model.add(Activation('sigmoid'))

    return dis_model


def build_adversarial_model(gen_model, dis_model):
    model = Sequential()
    model.add(gen_model)
    dis_model.trainable = False
    model.add(dis_model)
    return model


def calculate_inception_score(images_path, batch_size=1, splits=10):
    # Create an instance of InceptionV3
    model = InceptionResNetV2()

    images = None
    for image_ in glob.glob(images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if images is None:
            images = loaded_image
        else:
            images = np.concatenate([images, loaded_image], axis=0)

    # Calculate number of batches
    num_batches = (images.shape[0] + batch_size - 1) // batch_size

    probs = None

    # Use InceptionV3 to calculate probabilities
    for i in range(num_batches):
        image_batch = images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if probs is None:
            probs = prob
        else:
            probs = np.concatenate([prob, probs], axis=0)

    # Calculate Inception scores
    divs = []
    split_size = probs.shape[0] // splits

    for i in range(splits):
        prob_batch = probs[(i * split_size):((i + 1) * split_size), :]
        p_y = np.expand_dims(np.mean(prob_batch, 0), 0)
        div = prob_batch * (np.log(prob_batch / p_y))
        div = np.mean(np.sum(div, 1))
        divs.append(np.exp(div))

    return np.mean(divs), np.std(divs)


def calculate_mode_score(gen_images_path, real_images_path, batch_size=32, splits=10):
    # Create an instance of InceptionV3
    model = InceptionResNetV2()

    # Load real images
    real_images = None
    for image_ in glob.glob(real_images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if real_images is None:
            real_images = loaded_image
        else:
            real_images = np.concatenate([real_images, loaded_image], axis=0)

    # Load generated images
    gen_images = None
    for image_ in glob.glob(gen_images_path):
        # Load image
        loaded_image = image.load_img(image_, target_size=(299, 299))

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Another another dimension (Add batch dimension)
        loaded_image = np.expand_dims(loaded_image, axis=0)

        # Concatenate all images into one tensor
        if gen_images is None:
            gen_images = loaded_image
        else:
            gen_images = np.concatenate([gen_images, loaded_image], axis=0)

    # Calculate number of batches for generated images
    gen_num_batches = (gen_images.shape[0] + batch_size - 1) // batch_size
    gen_images_probs = None
    # Use InceptionV3 to calculate probabilities of generated images
    for i in range(gen_num_batches):
        image_batch = gen_images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if gen_images_probs is None:
            gen_images_probs = prob
        else:
            gen_images_probs = np.concatenate([prob, gen_images_probs], axis=0)

    # Calculate number of batches for real images
    real_num_batches = (real_images.shape[0] + batch_size - 1) // batch_size
    real_images_probs = None
    # Use InceptionV3 to calculate probabilities of real images
    for i in range(real_num_batches):
        image_batch = real_images[i * batch_size:(i + 1) * batch_size, :, :, :]
        prob = model.predict(preprocess_input(image_batch))

        if real_images_probs is None:
            real_images_probs = prob
        else:
            real_images_probs = np.concatenate([prob, real_images_probs], axis=0)

    # KL-Divergence: compute kl-divergence and mean of it
    num_gen_images = len(gen_images)
    split_scores = []

    for j in range(splits):
        gen_part = gen_images_probs[j * (num_gen_images // splits): (j + 1) * (num_gen_images // splits), :]
        real_part = real_images_probs[j * (num_gen_images // splits): (j + 1) * (num_gen_images // splits), :]
        gen_py = np.mean(gen_part, axis=0)
        real_py = np.mean(real_part, axis=0)
        scores = []
        for i in range(gen_part.shape[0]):
            scores.append(entropy(gen_part[i, :], gen_py))

        split_scores.append(np.exp(np.mean(scores) - entropy(gen_py, real_py)))

    final_mean = np.mean(split_scores)
    final_std = np.std(split_scores)

    return final_mean, final_std
