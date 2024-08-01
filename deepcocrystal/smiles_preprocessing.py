import re
from typing import List

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ELEMENTS_STR = r"(?<=\[)Cs(?=\])|Si|Xe|Ba|Rb|Ra|Sr|Dy|Li|Kr|Bi|Mn|He|Am|Pu|Cm|Pm|Ne|Th|Ni|Pr|Fe|Lu|Pa|Fm|Tm|Tb|Er|Be|Al|Gd|Eu|te|As|Pt|Lr|Sm|Ca|La|Ti|Te|Ac|Cf|Rf|Na|Cu|Au|Nd|Ag|Se|se|Zn|Mg|Br|Cl|Pb|U|V|K|C|B|H|N|O|S|P|F|I|b|c|n|o|s|p"
REGEX = (
    rf"(\[|\]|{ELEMENTS_STR}|" + r"\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%\d{2}|\d)"
)
RE_PATTERN = re.compile(REGEX)


def segment_smiles(smiles: str) -> List[str]:
    return RE_PATTERN.findall(smiles)


def space_separate_smiles(smiles: str) -> str:
    return " ".join(segment_smiles(smiles))


def space_separate_smiles_list(smiles_list: List[str]) -> np.array:
    return np.array([space_separate_smiles(smiles) for smiles in smiles_list])

def clean_smiles(
    smiles: str,
    uncharge=True,
    remove_stereochemistry=True,
    to_canonical=True,
):

    if remove_stereochemistry:
        smiles = drop_stereochemistry(smiles)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    uncharger = rdMolStandardize.Uncharger()
    if uncharge:
        mol = uncharger.uncharge(mol)
    return Chem.MolToSmiles(mol, canonical=to_canonical)


def clean_smiles_batch(
    smiles_batch: List[str],
    uncharge=True,
    remove_stereochemistry=True,
    to_canonical=True,
):
    return [
        clean_smiles(
            smiles,
            uncharge=uncharge,
            remove_stereochemistry=remove_stereochemistry,
            to_canonical=to_canonical,
        )
        for smiles in smiles_batch
    ]


def canonicalize(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return None


def canonicalize_batch(smiles_batch: List[str]) -> List[str]:
    return [canonicalize(smiles) for smiles in smiles_batch]


def uncharge(smiles: str, to_canonical=False) -> str:
    uncharger = rdMolStandardize.Uncharger()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid input chemical")
    return Chem.MolToSmiles(uncharger.uncharge(mol), canonical=to_canonical)


def uncharge_batch(smiles_batch: List[str], to_canonical=False) -> List[str]:
    uncharger = rdMolStandardize.Uncharger()
    uncharged_smiles = list()
    for smiles in smiles_batch:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES among inputs!")
        uncharged_smiles.append(
            Chem.MolToSmiles(uncharger.uncharge(mol), canonical=to_canonical)
        )
    return uncharged_smiles

def drop_stereochemistry(smiles: str):
    replace = {ord("/"): None, ord("\\"): None, ord("@"): None}
    return smiles.translate(replace)


def drop_stereochemistry_batch(smiles_batch: List[str]):
    replace = {ord("/"): None, ord("\\"): None, ord("@"): None}
    return [smiles.translate(replace) for smiles in smiles_batch]
