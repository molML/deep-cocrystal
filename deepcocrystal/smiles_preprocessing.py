import re
from typing import List

import numpy as np

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
