"""
NNDL EXPERIMENT 8: LeNet MNIST
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_8_lenet():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype("float32") / 255.0)[..., None]
    x_test = (x_test.astype("float32") / 255.0)[..., None]

    model = models.Sequential([
        layers.Input((28, 28, 1)),
        layers.Conv2D(6, 5, padding="same", activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Conv2D(16, 5, activation="tanh"),
        layers.AveragePooling2D(2),
        layers.Flatten(),
        layers.Dense(120, activation="tanh"),
        layers.Dense(84, activation="tanh"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=5, batch_size=128,
                        validation_split=0.1)
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Exp 8 test accuracy: {acc * 100:.2f}%")

    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Exp 8 - LeNet Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()


if __name__ == "__main__":
    experiment_8_lenet()
