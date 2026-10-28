# from keras_preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from keras.layers import (
    LSTM,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    Input,
    MaxPooling1D,
    TimeDistributed,
)
from keras.models import Model
from keras.utils import Progbar


def create_model(
    hyper: dict,
    word_emb,
    case_embeddings,
    phrase_embeddings,
    char_lu,
    labels_lu,
):
    # input layers
    words_input = Input(shape=(None,), dtype="int32", name="words_input")
    casing_input = Input(shape=(None,), dtype="int32", name="casing_input")
    phrasetag_input = Input(shape=(None,), dtype="int32", name="phrasetag_input")
    character_input = Input(
        shape=(None, 52),
        name="char_input",
    )

    # embedding layers
    words = Embedding(
        input_dim=word_emb.shape[0],
        output_dim=word_emb.shape[1],
        weights=[word_emb],
        trainable=False,
    )(words_input)
    caps = Embedding(
        output_dim=case_embeddings.shape[1],
        input_dim=case_embeddings.shape[0],
        weights=[case_embeddings],
        trainable=False,
    )(casing_input)

    phrasetag = Embedding(
        output_dim=phrase_embeddings.shape[1],
        input_dim=phrase_embeddings.shape[0],
        weights=[phrase_embeddings],
        trainable=False,
    )(phrasetag_input)

    # Char Convolution layers
    embed_char_out = TimeDistributed(
        Embedding(
            len(char_lu),
            hyper["FILTERS"],
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        ),
        name="char_embedding",
    )(character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(
        Conv1D(
            kernel_size=hyper["CONV_WIDTH"],
            filters=hyper["FILTERS"],
            padding="same",
            activation="tanh",
            strides=1,
        )
    )(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)

    chars = TimeDistributed(Flatten())(maxpool_out)
    chars = Dropout(0.5)(chars)

    output = Concatenate()([words, caps, chars, phrasetag])
    output = Bidirectional(
        LSTM(
            hyper["LSTM_STATE_SIZE"],
            return_sequences=True,
            dropout=hyper["LSTM_DROPOUT"],
            recurrent_dropout=hyper["REC_DROPOUT"],
        )
    )(output)
    output = TimeDistributed(Dense(len(labels_lu), activation="softmax"))(output)

    model = Model(
        inputs=[words_input, casing_input, character_input, phrasetag_input],
        outputs=[output],
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=hyper["OPTIMIZER"],
    )

    model.summary()

    return model


def create_model_conll2012(
    hyper: dict,
    word_emb,
    case_embeddings,
    char_lu,
    labels_lu,
):
    # input layers
    words_input = Input(shape=(None,), dtype="int32", name="words_input")
    casing_input = Input(shape=(None,), dtype="int32", name="casing_input")
    character_input = Input(
        shape=(None, 52),
        name="char_input",
    )

    # embedding layers
    words = Embedding(
        input_dim=word_emb.shape[0],
        output_dim=word_emb.shape[1],
        weights=[word_emb],
        trainable=False,
    )(words_input)
    caps = Embedding(
        output_dim=case_embeddings.shape[1],
        input_dim=case_embeddings.shape[0],
        weights=[case_embeddings],
        trainable=False,
    )(casing_input)

    # Char Convolution layers
    embed_char_out = TimeDistributed(
        Embedding(
            len(char_lu),
            hyper["FILTERS"],
            embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5),
        ),
        name="char_embedding",
    )(character_input)
    dropout = Dropout(0.5)(embed_char_out)
    conv1d_out = TimeDistributed(
        Conv1D(
            kernel_size=hyper["CONV_WIDTH"],
            filters=hyper["FILTERS"],
            padding="same",
            activation="tanh",
            strides=1,
        )
    )(dropout)
    maxpool_out = TimeDistributed(MaxPooling1D(52))(conv1d_out)

    chars = TimeDistributed(Flatten())(maxpool_out)
    chars = Dropout(0.5)(chars)

    output = Concatenate()([words, caps, chars])
    output = Bidirectional(
        LSTM(
            hyper["LSTM_STATE_SIZE"],
            return_sequences=True,
            dropout=hyper["LSTM_DROPOUT"],
            recurrent_dropout=hyper["REC_DROPOUT"],
        )
    )(output)
    output = TimeDistributed(Dense(len(labels_lu), activation="softmax"))(output)

    model = Model(
        inputs=[words_input, casing_input, character_input],
        outputs=[output],
    )
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=hyper["OPTIMIZER"],
    )

    model.summary()

    return model
