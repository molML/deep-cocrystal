import tensorflow as tf


class DeepCocrystal(tf.keras.Model):
    def __init__(
        self,
        n_cnn: int = 2,
        n_filters: int = 256,
        kernel_sizes: int = 4,
        cnn_activation: str = "selu",
        n_dense: int = 3,
        dense_layer_size: int = 1024,
        dense_activation: str = "relu",
        embedding_dim: int = 32,
        dropout_rate: float = 0.1,
        max_api_len: int = 80,
        max_cof_len: int = 80,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_cnn = n_cnn
        self.n_filters = n_filters
        self.kernel_sizes = kernel_sizes
        self.cnn_activation = cnn_activation
        self.n_dense = n_dense
        self.dense_layer_size = dense_layer_size
        self.dense_activation = dense_activation
        self.embedding_dim = embedding_dim
        self.dropout_rate = dropout_rate
        self.max_api_len = max_api_len
        self.max_cof_len = max_cof_len
        self.vocab_size = 32  # 30 elements + 1 for padding + 1 for unknown

        self.api_smiles_encoder = tf.keras.layers.TextVectorization(
            max_tokens=None,
            output_sequence_length=self.max_api_len,
            standardize=None,
            split="whitespace",
            output_mode="int",
            vocabulary="./data/vocabulary.txt",
            name="api_smiles_encoder",
        )
        self.cof_smiles_encoder = tf.keras.layers.TextVectorization(
            max_tokens=None,
            output_sequence_length=self.max_cof_len,
            standardize=None,
            split="whitespace",
            output_mode="int",
            vocabulary="./data/vocabulary.txt",
            name="cof_smiles_encoder",
        )

        self.api_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_api_len,
            mask_zero=True,
        )
        self.coformer_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size + 1,
            output_dim=self.embedding_dim,
            input_length=self.max_cof_len,
            mask_zero=True,
        )

        self.api_convs = [
            tf.keras.layers.Conv1D(
                filters=self.n_filters * (cnn_idx + 1),
                kernel_size=self.kernel_sizes,
                activation=self.cnn_activation,
                padding="valid",
                strides=1,
                name=f"api_cnn_{cnn_idx}",
            )
            for cnn_idx in range(self.n_cnn)
        ]
        self.cof_convs = [
            tf.keras.layers.Conv1D(
                filters=self.n_filters * (cnn_idx + 1),
                kernel_size=self.kernel_sizes,
                activation=self.cnn_activation,
                padding="valid",
                strides=1,
                name=f"cof_cnn_{cnn_idx}",
            )
            for cnn_idx in range(self.n_cnn)
        ]

        self.api_pooling = tf.keras.layers.GlobalMaxPooling1D()
        self.cof_pooling = tf.keras.layers.GlobalMaxPooling1D()

        self.interaction_concat = tf.keras.layers.Concatenate(axis=-1)

        self.interaction_denses = list()
        for dense_idx in range(self.n_dense):
            self.interaction_denses.append(
                tf.keras.layers.Dense(
                    self.dense_layer_size,
                    activation=self.dense_activation,
                    name=f"interaction_dense_{dense_idx}",
                )
            )
            self.interaction_denses.append(tf.keras.layers.Dropout(self.dropout_rate))

        self.prediction_head = tf.keras.layers.Dense(
            1, activation="sigmoid", name="prediction_head"
        )

    def call(self, inputs, **kwargs):
        api, cof = inputs
        api = self.api_smiles_encoder(api)
        cof = self.cof_smiles_encoder(cof)

        api = self.api_embedding(api)
        cof = self.coformer_embedding(cof)

        for api_conv, cof_conv in zip(self.api_convs, self.cof_convs):
            api = api_conv(api)
            cof = cof_conv(cof)

        api = self.api_pooling(api)
        cof = self.cof_pooling(cof)

        interaction = self.interaction_concat([api, cof])

        for dense in self.interaction_denses:
            interaction = dense(interaction)

        return self.prediction_head(interaction)
