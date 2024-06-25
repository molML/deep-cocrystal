import pandas as pd
import tensorflow as tf
from deepcocrystal import smiles_preprocessing
from deepcocrystal.deepcocrystal import DeepCocrystal

tf.keras.utils.set_random_seed(42)  # for reproducibility


def load_and_segment_api_coformer_pairs(path_to_csv: str):
    df = pd.read_csv(path_to_csv)
    api = df["SMILES1"].values
    coformer = df["SMILES2"].values
    labels = df["cocrystallization"].values

    # The DeepCocrystal model requires the SMILES strings to be space-separated
    api = smiles_preprocessing.space_separate_smiles_list(api)
    coformer = smiles_preprocessing.space_separate_smiles_list(coformer)

    return api, coformer, labels


train_api, train_coformer, train_labels = load_and_segment_api_coformer_pairs(
    "data/train.csv"
)
val_api, val_coformer, val_labels = load_and_segment_api_coformer_pairs("data/val.csv")

model = DeepCocrystal()  # will use the parameters in the paper by default
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    x=[train_api, train_coformer],
    y=train_labels,
    validation_data=([val_api, val_coformer], val_labels),
    batch_size=512,  # batch size used in the paper
    epochs=500,  # early stopping will stop training before this
    verbose=1,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=1e-5,
            patience=5,
            verbose=1,
            restore_best_weights=False,
        ),
        tf.keras.callbacks.CSVLogger(
            "training.log"
        ),  # save training history during training
    ],
)

model.save("deepcocrystal-trained")

loaded_model = tf.keras.models.load_model("deepcocrystal-trained")
test_api, test_coformer, test_labels = load_and_segment_api_coformer_pairs(
    "data/test_canonical.csv"
)

test_loss, test_accuracy = loaded_model.evaluate(
    x=[test_api, test_coformer],
    y=test_labels,
    batch_size=512,
    verbose=1,
)
