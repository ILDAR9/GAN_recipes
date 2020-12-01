import time
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.utils import to_categorical
from data import *
from model import *
from tqdm import tqdm

# Prevent allocate all the GPU memory
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

LOGDIR = "logs"
file_writer = tf.summary.create_file_writer(LOGDIR + "/metrics")
file_writer.set_as_default()


def write_log(name, value, batch_no):
    with file_writer.as_default():
        tf.summary.scalar(name, value, step=batch_no)
        file_writer.flush()


def save_rgb_img(img, path):
    """
    Save an rgb image
    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img)
    ax.axis("off")
    ax.set_title("Image")

    plt.savefig(path)
    plt.close()


if __name__ == '__main__':
    # Define hyperparameters
    wiki_dir = "wiki_crop"
    epochs = 500
    batch_size = 84
    image_shape = (64, 64, 3)
    z_shape = 100
    TRAIN_GAN = True
    TRAIN_ENCODER = False
    TRAIN_GAN_WITH_FR = False
    fr_image_shape = (192, 192, 3)
    N = 45

    # Define optimizers
    dis_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    gen_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)
    adversarial_optimizer = Adam(lr=0.0002, beta_1=0.5, beta_2=0.999, epsilon=10e-8)

    result_out = "results"
    if not os.path.exists(result_out):
        os.makedirs(result_out)

    """
    Build and compile networks
    """
    # Build and compile the discriminator network
    discriminator = build_discriminator()
    discriminator.compile(loss=['binary_crossentropy'], optimizer=dis_optimizer)

    # Build and compile the generator network
    generator = build_generator()
    generator.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    # Build and compile the adversarial model
    discriminator.trainable = False
    input_z_noise = Input(shape=(100,))
    input_label = Input(shape=(6,))
    recons_images = generator([input_z_noise, input_label])
    valid = discriminator([recons_images, input_label])
    adversarial_model = Model(inputs=[input_z_noise, input_label], outputs=[valid])
    adversarial_model.compile(loss=['binary_crossentropy'], optimizer=gen_optimizer)

    tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
    tensorboard.set_model(generator)
    tensorboard.set_model(discriminator)

    """
    Load the dataset
    """
    images, age_list = load_data(wiki_dir=wiki_dir, dataset="wiki")
    age_cat = age_to_category(age_list)
    final_age_cat = np.reshape(np.array(age_cat), [len(age_cat), 1])
    classes = len(set(age_cat))
    y = to_categorical(final_age_cat, num_classes=len(set(age_cat)))
    data_gen = lambda: images_gen(wiki_dir, images, (image_shape[0], image_shape[1]))
    # loaded_images = load_images(wiki_dir, images, (image_shape[0], image_shape[1]))
    # Implement label smoothing
    real_labels = np.ones((batch_size, 1), dtype=np.float32) * 0.9
    fake_labels = np.zeros((batch_size, 1), dtype=np.float32) * 0.1

    """
    Train the generator and the discriminator network
    """
    if TRAIN_GAN:

        # Load the generator network's weights
        if N > 0:
            generator.load_weights(f"generator_{N}.h5")
            discriminator.load_weights(f"discriminator_{N}.h5")
            print(f"Weights are loaded for epoch {N}")

        for epoch in tqdm(range(N+1, epochs), initial=N+1, total=epochs):
            print("Epoch:{}".format(epoch))
            loaded_images = tf.data.Dataset.from_generator(data_gen, output_types=tf.float32, ). \
                batch(batch_size, drop_remainder=True). \
                map(lambda x: x / 127.5 - 1.0). \
                prefetch(buffer_size=batch_size * 10)

            gen_losses = []
            dis_losses = []

            # number_of_batches = len(loaded_images) // batch_size
            # print("Number of batches:", number_of_batches)
            # for index in range(number_of_batches):
            for index, images_batch in enumerate(loaded_images):
                y_batch = y[index * batch_size:(index + 1) * batch_size]
                if y_batch.shape[0] < batch_size:
                    print(f"pass Batch:{index + 1}")
                    continue
                # print(f"Batch:{index + 1}")

                # images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                # images_batch = images_batch / 127.5 - 1.0
                # images_batch = images_batch.astype(np.float32)

                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                """
                Train the discriminator network
                """

                # Generate fake images
                initial_recon_images = generator.predict_on_batch([z_noise, y_batch])

                d_loss_real = discriminator.train_on_batch([images_batch, y_batch], real_labels)
                d_loss_fake = discriminator.train_on_batch([initial_recon_images, y_batch], fake_labels)

                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                """
                Train the generator network
                """

                z_noise2 = np.random.normal(0, 1, size=(batch_size, z_shape))
                random_labels = np.random.randint(0, 6, batch_size).reshape(-1, 1)
                random_labels = to_categorical(random_labels, 6)

                g_loss = adversarial_model.train_on_batch([z_noise2, random_labels], np.array([1] * batch_size).reshape((-1, 1)))
                if index % 5 == 0:
                    print(f"d_loss:{d_loss}, g_loss:{g_loss}")

                gen_losses.append(g_loss)
                dis_losses.append(d_loss)

            # Write losses to Tensorboard
            write_log('g_loss', np.mean(gen_losses), epoch)
            write_log('d_loss', np.mean(dis_losses), epoch)

            """
            Generate images after every n-th epoch
            """
            if epoch % 1 == 0:
                print(f"Gen results for epocj {epoch}")
                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path=f"{result_out}/img_{epoch}_{i}.png")
                # Save networks
                try:
                    generator.save_weights(f"generator_{epoch}.h5")
                    discriminator.save_weights(f"discriminator_{epoch}.h5")
                except Exception as e:
                    print("Error:", e)
                    print("Error:", e)

        # Save networks
        try:
            generator.save_weights("generator.h5")
            discriminator.save_weights("discriminator.h5")
        except Exception as e:
            print("Error:", e)
            print("Error:", e)

    """
    Train encoder
    """

    if TRAIN_ENCODER:
        # Build and compile encoder
        encoder = build_encoder()
        encoder.compile(loss=euclidean_distance_loss, optimizer='adam')

        # Load the generator network's weights
        try:
            generator.load_weights("generator.h5")
        except Exception as e:
            print("Error:", e)

        z_i = np.random.normal(0, 1, size=(5000, z_shape))

        y = np.random.randint(low=0, high=6, size=(5000,), dtype=np.int64)
        num_classes = len(set(y))
        y = np.reshape(np.array(y), [len(y), 1])
        y = to_categorical(y, num_classes=num_classes)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            encoder_losses = []

            number_of_batches = int(z_i.shape[0] / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                z_batch = z_i[index * batch_size:(index + 1) * batch_size]
                y_batch = y[index * batch_size:(index + 1) * batch_size]

                generated_images = generator.predict_on_batch([z_batch, y_batch])

                # Train the encoder model
                encoder_loss = encoder.train_on_batch(generated_images, z_batch)
                print("Encoder loss:", encoder_loss)

                encoder_losses.append(encoder_loss)

            # Write the encoder loss to Tensorboard
            write_log("encoder_loss", np.mean(encoder_losses), epoch)

        # Save the encoder model
        encoder.save_weights("encoder.h5")

    """
    Optimize the encoder and the generator network
    """
    if TRAIN_GAN_WITH_FR:

        # Load the encoder network
        encoder = build_encoder()
        encoder.load_weights("encoder.h5")

        # Load the generator network
        generator.load_weights("generator.h5")

        image_resizer = build_image_resizer()
        image_resizer.compile(loss=['binary_crossentropy'], optimizer='adam')

        # Face recognition model
        fr_model = build_fr_model(input_shape=fr_image_shape)
        fr_model.compile(loss=['binary_crossentropy'], optimizer="adam")

        # Make the face recognition network as non-trainable
        fr_model.trainable = False

        # Input layers
        input_image = Input(shape=(64, 64, 3))
        input_label = Input(shape=(6,))

        # Use the encoder and the generator network
        latent0 = encoder(input_image)
        gen_images = generator([latent0, input_label])

        # Resize images to the desired shape
        resized_images = Lambda(lambda x: K.resize_images(gen_images, height_factor=3, width_factor=3,
                                                          data_format='channels_last'))(gen_images)
        embeddings = fr_model(resized_images)

        # Create a Keras model and specify the inputs and outputs for the network
        fr_adversarial_model = Model(inputs=[input_image, input_label], outputs=[embeddings])

        # Compile the model
        fr_adversarial_model.compile(loss=euclidean_distance_loss, optimizer=adversarial_optimizer)

        for epoch in range(epochs):
            print("Epoch:", epoch)

            reconstruction_losses = []

            number_of_batches = int(len(loaded_images) / batch_size)
            print("Number of batches:", number_of_batches)
            for index in range(number_of_batches):
                print("Batch:", index + 1)

                images_batch = loaded_images[index * batch_size:(index + 1) * batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[index * batch_size:(index + 1) * batch_size]

                images_batch_resized = image_resizer.predict_on_batch(images_batch)

                real_embeddings = fr_model.predict_on_batch(images_batch_resized)

                reconstruction_loss = fr_adversarial_model.train_on_batch([images_batch, y_batch], real_embeddings)

                print("Reconstruction loss:", reconstruction_loss)

                reconstruction_losses.append(reconstruction_loss)

            # Write the reconstruction loss to Tensorboard
            write_log("reconstruction_loss", np.mean(reconstruction_losses), epoch)

            """
            Generate images
            """
            if epoch % 10 == 0:
                images_batch = loaded_images[0:batch_size]
                images_batch = images_batch / 127.5 - 1.0
                images_batch = images_batch.astype(np.float32)

                y_batch = y[0:batch_size]
                z_noise = np.random.normal(0, 1, size=(batch_size, z_shape))

                gen_images = generator.predict_on_batch([z_noise, y_batch])

                for i, img in enumerate(gen_images[:5]):
                    save_rgb_img(img, path=f"{result_out}/img_opt_{epoch}_{i}.png")

        # Save improved weights for both of the networks
        generator.save_weights("generator_optimized.h5")
        encoder.save_weights("encoder_optimized.h5")
