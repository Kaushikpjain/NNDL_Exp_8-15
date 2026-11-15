"""
NNDL EXPERIMENT 15: DCGAN MNIST
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_15_dcgan():
    import tensorflow as tf
    from tensorflow.keras import layers

    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") - 127.5) / 127.5
    x_train = x_train[..., None]
    batch_size, latent_dim = 128, 100
    dataset = (tf.data.Dataset.from_tensor_slices(x_train)
               .shuffle(60000).batch(batch_size, drop_remainder=True))

    generator = tf.keras.Sequential([
        layers.Input((latent_dim,)),
        layers.Dense(7 * 7 * 128, use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Reshape((7, 7, 128)),
        layers.Conv2DTranspose(64, 4, strides=2, padding="same",
                               use_bias=False),
        layers.BatchNormalization(), layers.ReLU(),
        layers.Conv2DTranspose(1, 4, strides=2, padding="same",
                               activation="tanh")
    ])
    discriminator = tf.keras.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2), layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(negative_slope=0.2), layers.Dropout(0.3),
        layers.Flatten(), layers.Dense(1, activation="sigmoid")
    ])

    bce = tf.keras.losses.BinaryCrossentropy()
    g_opt = tf.keras.optimizers.Adam(1e-4)
    d_opt = tf.keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real):
        n = tf.shape(real)[0]
        noise = tf.random.normal([n, latent_dim])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = generator(noise, training=True)
            real_score = discriminator(real, training=True)
            fake_score = discriminator(fake, training=True)
            # Non-saturating generator objective.
            g_loss = bce(tf.ones_like(fake_score), fake_score)
            d_loss = (bce(tf.ones_like(real_score), real_score)
                      + bce(tf.zeros_like(fake_score), fake_score))
        g_grad = gt.gradient(g_loss, generator.trainable_variables)
        d_grad = dt.gradient(d_loss, discriminator.trainable_variables)
        g_opt.apply_gradients(zip(g_grad, generator.trainable_variables))
        d_opt.apply_gradients(zip(d_grad, discriminator.trainable_variables))
        return g_loss, d_loss

    def show_samples(epoch):
        noise = tf.random.normal([16, latent_dim])
        images = (generator(noise, training=False) + 1.0) / 2.0
        plt.figure(figsize=(4, 4))
        for i in range(16):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i, :, :, 0], cmap="gray")
            plt.axis("off")
        plt.suptitle(f"Experiment 15 - Generated MNIST, epoch {epoch}")
        plt.tight_layout()
        plt.show()

    for epoch in range(1, 11):
        g_losses, d_losses = [], []
        for real_batch in dataset:
            gl, dl = train_step(real_batch)
            g_losses.append(float(gl))
            d_losses.append(float(dl))
        print(f"Epoch {epoch}: G loss={np.mean(g_losses):.4f}, "
              f"D loss={np.mean(d_losses):.4f}")
        if epoch == 1 or epoch % 2 == 0:
            show_samples(epoch)


if __name__ == "__main__":
    experiment_15_dcgan()
