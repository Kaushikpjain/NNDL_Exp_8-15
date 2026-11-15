"""
NNDL EXPERIMENT 12: NER BiLSTM
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_12_ner_bilstm():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from sklearn.metrics import classification_report

    sentences = [
        ["John", "lives", "in", "London"], ["Mary", "works", "at", "Google"],
        ["Alice", "visits", "Paris"], ["Google", "is", "in", "California"],
        ["David", "works", "at", "Microsoft"], ["Ravi", "lives", "in", "Chennai"],
        ["Amazon", "is", "a", "company"], ["Sarah", "visits", "New", "York"]
    ]
    tags = [
        ["B-PER", "O", "O", "B-LOC"], ["B-PER", "O", "O", "B-ORG"],
        ["B-PER", "O", "B-LOC"], ["B-ORG", "O", "O", "B-LOC"],
        ["B-PER", "O", "O", "B-ORG"], ["B-PER", "O", "O", "B-LOC"],
        ["B-ORG", "O", "O", "O"], ["B-PER", "O", "B-LOC", "I-LOC"]
    ]
    words = sorted({w for s in sentences for w in s})
    tag_set = sorted({t for row in tags for t in row})
    word_to_id = {w: i + 1 for i, w in enumerate(words)}
    tag_to_id = {t: i for i, t in enumerate(tag_set)}
    id_to_tag = {i: t for t, i in tag_to_id.items()}

    X = [[word_to_id[w] for w in s] for s in sentences]
    Y = [[tag_to_id[t] for t in row] for row in tags]
    max_len = 5
    X = pad_sequences(X, maxlen=max_len, padding="post", value=0)
    Y = pad_sequences(Y, maxlen=max_len, padding="post",
                      value=tag_to_id["O"])
    Y_hot = tf.keras.utils.to_categorical(Y, num_classes=len(tag_set))

    model = models.Sequential([
        layers.Input((max_len,)),
        layers.Embedding(len(word_to_id) + 1, 32, mask_zero=True),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.TimeDistributed(layers.Dense(len(tag_set), activation="softmax"))
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit(X, Y_hot, epochs=30, batch_size=2)

    pred = np.argmax(model.predict(X, verbose=0), axis=-1)
    actual = np.argmax(Y_hot, axis=-1)
    true_tags, pred_tags = [], []
    for i in range(len(X)):
        for j in range(max_len):
            if X[i, j] != 0:
                true_tags.append(id_to_tag[actual[i, j]])
                pred_tags.append(id_to_tag[pred[i, j]])
    print(classification_report(true_tags, pred_tags, zero_division=0))
    print("Tiny demonstration dataset: metrics are illustrative only.")


if __name__ == "__main__":
    experiment_12_ner_bilstm()
