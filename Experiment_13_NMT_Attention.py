"""
NNDL EXPERIMENT 13: NMT Attention
This file is fully standalone. Run it directly.

Install dependencies:
    pip install tensorflow scikit-learn matplotlib numpy
Some experiments may use only a subset of these packages.
"""

def experiment_13_nmt_attention():
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    import re

    pairs = [
        ("hello", "bonjour"), ("good morning", "bonjour"),
        ("thank you", "merci"), ("good night", "bonne nuit"),
        ("i am happy", "je suis heureux"),
        ("how are you", "comment allez vous"),
        ("i love you", "je t aime"), ("see you soon", "a bientot"),
        ("good evening", "bonsoir"), ("i am a student", "je suis etudiant")
    ]
    tokenize = lambda s: re.findall(r"\w+", s.lower())
    src = [tokenize(a) for a, _ in pairs]
    tgt = [["<start>"] + tokenize(b) + ["<end>"] for _, b in pairs]
    src_words = sorted({w for s in src for w in s})
    tgt_words = sorted({w for s in tgt for w in s} | {"<pad>", "<unk>"})
    src_vocab = {w: i + 1 for i, w in enumerate(src_words)}
    tgt_vocab = {w: i for i, w in enumerate(tgt_words)}

    max_src = max(map(len, src))
    max_tgt = max(map(len, tgt))
    X = pad_sequences([[src_vocab[w] for w in s] for s in src],
                      maxlen=max_src, padding="post")
    Y = pad_sequences([[tgt_vocab[w] for w in s] for s in tgt],
                      maxlen=max_tgt, padding="post",
                      value=tgt_vocab["<pad>"])
    dec_in, dec_target = Y[:, :-1], Y[:, 1:]

    enc_in = layers.Input((max_src,))
    enc_emb = layers.Embedding(len(src_vocab) + 1, 64,
                               mask_zero=True)(enc_in)
    enc_out = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True)
    )(enc_emb)

    dec_in_layer = layers.Input((max_tgt - 1,))
    dec_emb = layers.Embedding(len(tgt_vocab), 128,
                               mask_zero=True)(dec_in_layer)
    dec_out = layers.LSTM(128, return_sequences=True)(dec_emb)

    context = layers.AdditiveAttention()([dec_out, enc_out])
    combined = layers.Concatenate()([dec_out, context])
    output = layers.Dense(len(tgt_vocab), activation="softmax")(combined)

    model = models.Model([enc_in, dec_in_layer], output)
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.fit([X, dec_in], dec_target, epochs=100, batch_size=2)
    print("Experiment 13 NMT model trained.")
    print("This tiny corpus demonstrates the architecture, not reliable translation.")


if __name__ == "__main__":
    experiment_13_nmt_attention()
