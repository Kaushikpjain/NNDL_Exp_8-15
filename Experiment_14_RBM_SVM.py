"""
NNDL EXPERIMENT 14: RBM SVM
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_14_rbm_svm():
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import BernoulliRBM
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score, classification_report

    digits = load_digits()
    X, y = digits.data / 16.0, digits.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = Pipeline([
        ("rbm", BernoulliRBM(n_components=64, learning_rate=0.05,
                             n_iter=20, random_state=42)),
        ("svm", SVC(kernel="linear"))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"Experiment 14 accuracy: {accuracy_score(y_test, pred) * 100:.2f}%")
    print(classification_report(y_test, pred, zero_division=0))

    filters = model.named_steps["rbm"].components_
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filters[i].reshape(8, 8), cmap="gray")
        ax.axis("off")
    plt.suptitle("Experiment 14 - RBM Learned Filters")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    experiment_14_rbm_svm()
