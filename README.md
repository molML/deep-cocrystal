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

## Data Preparation and Model Training :pill:

Before using DeepCocrystal, you need to format your data into a `.csv` file with three columns: `SMILES1`, `SMILES2`, and **cocrystallization**, which contains the SMILES of the active pharmaceutical ingredients (APIs), coformers, and the observed cocrystallization output, respectively. You can see [train.csv](https://github.com/molML/deep-cocrystal/blob/main/data/train.csv) for an example.

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

Afterward, you can use [the code under the examples folder](https://github.com/molML/deep-cocrystal/blob/main/examples/train_and_predict.py) to train your model! :rocket:

### SMILES Randomization :twisted_rightwards_arrows:

Starting from canonical SMILES, you can perform SMILES randomization on your data, using the code provided [here](https://github.com/EBjerrum/SMILES-enumeration.git). [We](https://chemrxiv.org/engage/chemrxiv/article-details/66704f0501103d79c56770b2) observed SMILES randomization to strengthen the generalizability of the models and help estimate the prediction uncertainty.


## Prediction :crystal_ball:

Now that you have your model (or you can use the DeepCocrystal model available here), you can predict the co-crystallization of any API-coformer pair model. How? Like this:

```python
from deepcocrystal import smiles_preprocessing

# load the model
loaded_model = tf.keras.models.load_model("deepcocrystal-trained")

# load the data
test_api = smiles_preprocessing.space_separate_smiles_list(apis)
test_coformer = smiles_preprocessing.space_separate_smiles_list(coformers)

# predict
predictions = loaded_model.predict(
    x=[test_api, test_coformer], 
    batch_size = 512,
    verbose = 1
    )
```

Voila! You have your co-crystallization predictions  :tada: 

> [!IMPORTANT]
> The `deepcocrystal-trained` model provided here is trained only on non-proprietary data to allow a permissive license (560 less API-coformer pairs). The accuracy of this model is 75% and the sensitivity is 70%.


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