"""
NNDL EXPERIMENT 11: Character Level LSTM
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_11_character_lstm():
    import tensorflow as tf
    from tensorflow.keras import layers, models

    text = ("machine learning is an important field. "
            "deep learning uses neural networks. "
            "artificial intelligence helps machines learn. "
            "machine learning allows computers to learn patterns.").lower()
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    seq_len = 20
    X, y = [], []
    for i in range(len(text) - seq_len):
        X.append([char_to_idx[c] for c in text[i:i + seq_len]])
        y.append(char_to_idx[text[i + seq_len]])
    X, y = np.array(X), np.array(y)

    model = models.Sequential([
        layers.Input((seq_len,)),
        layers.Embedding(len(chars), 32),
        layers.LSTM(64),
        layers.Dense(len(chars), activation="softmax")
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X, y, epochs=20, batch_size=16)

    def generate(seed="machine learning", length=100, temperature=0.8):
        generated = "".join(c if c in char_to_idx else " " for c in seed.lower())
        for _ in range(length):
            seq = generated[-seq_len:].rjust(seq_len)
            encoded = np.array([[char_to_idx.get(c, char_to_idx[" "])
                                 for c in seq]])
            probs = model.predict(encoded, verbose=0)[0]
            probs = np.log(probs + 1e-8) / temperature
            probs = np.exp(probs - np.max(probs))
            probs /= probs.sum()
            next_id = np.random.choice(len(chars), p=probs)
            generated += idx_to_char[next_id]
        return generated

    print("\nGenerated text:\n", generate())


if __name__ == "__main__":
    experiment_11_character_lstm()
