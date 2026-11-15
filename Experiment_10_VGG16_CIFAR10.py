"""
NNDL EXPERIMENT 10: VGG16 CIFAR10
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_10_vgg16():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.applications.vgg16 import preprocess_input

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = x_train[:2000], y_train[:2000].ravel()
    x_test, y_test = x_test[:500], y_test[:500].ravel()

    base = VGG16(weights="imagenet", include_top=False,
                 input_shape=(224, 224, 3))
    base.trainable = False

    def extract(images):
        ds = tf.data.Dataset.from_tensor_slices(images).batch(32)
        out = []
        for batch in ds:
            batch = tf.image.resize(tf.cast(batch, tf.float32), (224, 224))
            out.append(base(preprocess_input(batch), training=False).numpy())
        return np.concatenate(out, axis=0)

    print("Extracting VGG16 features...")
    train_features, test_features = extract(x_train), extract(x_test)
    model = models.Sequential([
        layers.Input(train_features.shape[1:]),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_features, y_train, epochs=10, batch_size=32,
                        validation_split=0.2)
    _, acc = model.evaluate(test_features, y_test, verbose=0)
    print(f"Exp 10 test accuracy: {acc * 100:.2f}%")
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Exp 10 - VGG16 Transfer Learning")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()


if __name__ == "__main__":
    experiment_10_vgg16()
