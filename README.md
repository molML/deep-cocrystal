# DeepCocrystal

Welcome to [DeepCocrystal](https://chemrxiv.org/engage/chemrxiv/article-details/66704f0501103d79c56770b2), a predictive model that will help you select promising coformers for your co-crystallization trails, simply from the SMILES of your molecules. 

Let's get started!  :sparkles:

## Installation :hammer_and_wrench:

You first need to download this codebase. You can either click on the green button on the top-right corner of this page and download the codebase as a zip file or clone the repository with the following command, if you have git installed:

```bash
git clone https://github.com/molML/deep-cocrystal.git
```

We'll use `conda` to create a new environment for our codebase. If you haven't used conda before, we recommend you take a look at [this tutorial](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) before moving forward.


Otherwise, fire up a terminal *in the (root) directory of the codebase* and type the following commands:

```bash
conda create -n deepcocrystal_env python==3.10.13 
conda activate deepcocrystal_env 
conda install --file requirements.txt -c conda-forge  
python -m pip install .  # install this codebase -- make sure that you are in the root directory of the codebase
```

That's it! You have successfully installed our codebase.

## Choose your molecules and prepare your data :thought_balloon:  :pill:

Before using DeepCocrystal, you need to prepare your data. Choose the APIs and coformers for which you want to predict co-crystallization. You have two options for model input:

"1." CANONICAL SMILES :full_moon_with_face:

```python
from preprocessing import clean_smiles_batch


#Import your dataset
df = pd.read_csv("./data/test.csv") #your file directory 
apis = df["SMILES1"].values.tolist()
coformers = df["SMILES2"].values.tolist()

# Clean and canonicalize SMILES of API
apis = clean_smiles_batch(
    apis,
    uncharge=True,
    remove_stereochemistry=True,
    to_canonical=True,
)
# Clean and canonicalize SMILES of coformers 
coformers = clean_smiles_batch(
    coformers,
    uncharge=True,
    remove_stereochemistry=True,
    to_canonical=True,
)
```

"2." RANDOMIZED SMILES :twisted_rightwards_arrows:

Starting from canonical SMILES, you can perform 10-fold SMILES randomization for each molecular pair in your test set, using the code provided [here](https://github.com/EBjerrum/SMILES-enumeration.git). This step will allow you to estimate the uncertainty of the predictions. For further details, please refer to our [paper](https://chemrxiv.org/engage/chemrxiv/article-details/66704f0501103d79c56770b2).


## Predict co-crystallization with DeepCocrystal :crystal_ball:

Now that your data are ready, you can predict their co-crystallization by loading our pre-trained model. How? Use the load_and_predict.py script or the following code lines: 

```python
from deepcocrystal import smiles_preprocessing

#pre-trained model loading
loaded_model = tf.keras.models.load_model("deepcocrystal-trained")

test_api = smiles_preprocessing.space_separate_smiles_list(apis)
test_coformer = smiles_preprocessing.space_separate_smiles_list(coformers)

#DeepCocrystal prediction
predictions = loaded_model.predict(
    x=[test_api, test_coformer], 
    batch_size = 512,
    verbose = 1
    )
```

Voila! :tada: You have successfully predict the co-crystallization of your API-coformer pairs!

## Additional Informations :warning:

The 'deepcocrystal-trained' model provided here was trained on a smaller dataset compared to the model described in the article to avoid disclosing sensitive industrial data. Consequently, the accuracy (full dataset= 78%, pubblic dataset= 75%) and sensitivity (full dataset= 81%, pubblic dataset= 70%) of this model are slightly lower than those reported in the paper (sorry about that). If you have additional data and wish to retrain DeepCocrystal, you can use the train_and_predict.py file in the examples. All you need are the SMILES of the two molecules and the co-crystallization outcome (see train.csv in the data folder).


##  Closing Remarks :fireworks: 

Thanks again for finding our code interesting! Please consider starring the repository :sparkles: and citing our work if this codebase has been useful for driving your experimental tests :woman_scientist: :man_scientist: 


```bibtex
@article{birolo2024deep,
  title={Deep Supramolecular Language Processing for Co-crystal Prediction},
  author={Birolo, Rebecca and {\"O}z{\c{c}}elik, R{\i}za and Aramini, Andrea and Gobetto, Roberto and Chierotti, Michele Remo and Grisoni, Francesca},
  year={2024}
}
```

If you have any questions, please don't hesitate to open an issue in this repository. We'll be happy to help :man_dancing: 

Hope to see you around! :wave: :wave: :wave: