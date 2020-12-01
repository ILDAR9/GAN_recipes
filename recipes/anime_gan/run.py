import time
import keras.backend as K
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from imageio import imread
from keras.callbacks import TensorBoard
from keras.optimizers import SGD
from tqdm import tqdm
from models import *

# Prevent allocate all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
#
# K.image_data_format('tf')

np.random.seed(1337)

LOGDIR = "logs"
file_writer = tf.summary.create_file_writer(LOGDIR + "/metrics")
file_writer.set_as_default()


def write_log(name, value, batch_no):
    with file_writer.as_default():
        tf.summary.scalar(name, value, step=batch_no)
        file_writer.flush()


def denormalize(img):
    img = (img + 1) * 127.5
    return img.astype(np.uint8)


def normalize(img):
    return (img - 127.5) / 127.5


def visualize_rgb(img):
    """
    Visualize a rgb image
    :param img: RGB image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")
    plt.show()


def read_image(image_path, target_size=(64, 64)):
    try:
        # Load image
        loaded_image = image.load_img(image_path, target_size=target_size)

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Add another dimension (Add batch dimension)
        # loaded_image = np.expand_dims(loaded_image, axis=0)

        return loaded_image
    except Exception as e:
        print("Error:", e)


def save_rgb_img(img, path):
    """
    Save a rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("RGB Image")

    plt.savefig(path)
    plt.close()


def read_image(image_path, target_size=(64, 64)):
    try:
        # Load image
        loaded_image = image.load_img(image_path, target_size=target_size)

        # Convert PIL image to numpy ndarray
        loaded_image = image.img_to_array(loaded_image)

        # Add another dimension (Add batch dimension)
        # loaded_image = np.expand_dims(loaded_image, axis=0)

        return loaded_image.astype(float)
    except Exception as e:
        print("Error:", e)


def train():
    start_time = time.time()
    dataset_listing = "data/listing.txt"
    img_out = "results"
    batch_size = 128
    z_shape = 100
    epochs = 10000
    dis_learning_rate = 0.005
    gen_learning_rate = 0.005
    dis_momentum = 0.5
    gen_momentum = 0.5
    dis_nesterov = True
    gen_nesterov = True

    dis_optimizer = SGD(lr=dis_learning_rate, momentum=dis_momentum, nesterov=dis_nesterov)
    gen_optimizer = SGD(lr=gen_learning_rate, momentum=gen_momentum, nesterov=gen_nesterov)

    if not os.path.exists(img_out):
        os.makedirs(img_out)

    # Load images
    all_images = []
    with open(dataset_listing, 'r') as f:
        fnames = f.readlines()
    for fname in tqdm(fnames):
        fname = os.path.join("data", fname.strip())
        im = read_image(fname)
        if im is not None:
            all_images.append(im)
    del fnames

    X = np.array(all_images)
    X = (X - 127.5) / 127.5

    dis_model = build_discriminator()
    dis_model.compile(loss='binary_crossentropy', optimizer=dis_optimizer)

    gen_model = build_generator()
    gen_model.compile(loss='mse', optimizer=gen_optimizer)

    adversarial_model = build_adversarial_model(gen_model, dis_model)
    adversarial_model.compile(loss='binary_crossentropy', optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir=f"{LOGDIR}/{time.time()}", write_images=True, write_grads=True, write_graph=True)
    tensorboard.set_model(gen_model)
    tensorboard.set_model(dis_model)

    for epoch in range(epochs):
        print("--------------------------")
        print("Epoch:", epoch)

        dis_losses = []
        gen_losses = []

        num_batches = int(X.shape[0] / batch_size)

        print("Number of batches:", num_batches)
        for index in range(num_batches):
            print("Batch:{}", index)

            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            generated_images = gen_model.predict_on_batch(z_noise)

            # visualize_rgb(generated_images[0])

            """
            Train the discriminator model
            """

            dis_model.trainable = True

            image_batch = X[index * batch_size:(index + 1) * batch_size]

            y_real = np.ones((batch_size,)) * 0.9
            y_fake = np.zeros((batch_size,)) * 0.1

            dis_loss_real = dis_model.train_on_batch(image_batch, y_real)
            dis_loss_fake = dis_model.train_on_batch(generated_images, y_fake)

            d_loss = (dis_loss_real + dis_loss_fake) / 2
            print("d_loss:", d_loss)

            dis_model.trainable = False

            """
            Train the generator model(adversarial model)
            """
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            # z_noise = np.random.uniform(-1, 1, size=(batch_size, 100))

            g_loss = adversarial_model.train_on_batch(z_noise, y_real)
            print("g_loss:", g_loss)

            dis_losses.append(d_loss)
            gen_losses.append(g_loss)

        """
        Sample some images and save them
        """
        if epoch % 100 == 0:
            z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))
            gen_images1 = gen_model.predict_on_batch(z_noise)

            for img in gen_images1[:2]:
                save_rgb_img(img, f"{img_out}/one_{epoch}.png")

        print(f"Epoch:{epoch}, dis_loss:{np.mean(dis_losses)}")
        print(f"Epoch:{epoch}, gen_loss: {np.mean(gen_losses)}")

        """
        Save losses to Tensorboard after each epoch
        """
        write_log('discriminator_loss', np.mean(dis_losses), epoch)
        write_log('generator_loss', np.mean(gen_losses), epoch)

    """
    Save models
    """
    gen_model.save("generator_model.h5")
    dis_model.save("generator_model.h5")

    print(f"Time: {time.time() - start_time} seconds")


if __name__ == '__main__':
    train()
