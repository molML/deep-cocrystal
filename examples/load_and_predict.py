import pandas as pd
import tensorflow as tf
from deepcocrystal import smiles_preprocessing

loaded_model = tf.keras.models.load_model("deepcocrystal-trained")

df_test = pd.read_csv("./data/test_canonical.csv")
test_api = df_test["SMILES1"].values
test_coformer = df_test["SMILES2"].values
test_labels = df_test["cocrystallization"].values

test_api = smiles_preprocessing.space_separate_smiles_list(test_api)
test_coformer = smiles_preprocessing.space_separate_smiles_list(test_coformer)

test_loss, test_accuracy = loaded_model.evaluate(
    x=[test_api, test_coformer],
    y=test_labels,
    batch_size=512,
    verbose=1,
)