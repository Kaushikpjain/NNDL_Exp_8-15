"""
NNDL EXPERIMENT 9: ResNet50 CIFAR10
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_9_resnet():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.applications.resnet50 import preprocess_input

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, y_train = x_train[:3000], y_train[:3000].ravel()
    x_test, y_test = x_test[:500], y_test[:500].ravel()

    def make_ds(images, labels):
        ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(32)
        def prep(x, y):
            x = tf.image.resize(tf.cast(x, tf.float32), (224, 224))
            return preprocess_input(x), y
        return ds.map(prep).prefetch(tf.data.AUTOTUNE)

    train_ds, test_ds = make_ds(x_train, y_train), make_ds(x_test, y_test)
    base = ResNet50(weights="imagenet", include_top=False,
                    input_shape=(224, 224, 3))
    base.trainable = False
    model = models.Sequential([
        layers.Input((224, 224, 3)), base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    history = model.fit(train_ds, epochs=3, validation_data=test_ds)
    _, acc = model.evaluate(test_ds, verbose=0)
    print(f"Exp 9 test accuracy: {acc * 100:.2f}%")
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Validation")
    plt.title("Exp 9 - ResNet50 Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.show()
    print("This code performs classification, not bounding-box object detection.")


if __name__ == "__main__":
    experiment_9_resnet()
