from rdkit.Chem import GetSSSR, MolFromSmiles
from baybe.utils.dataframe import df_drop_single_value_columns
import pandas as pd


def getMolFromSmile(x, sanitize=True):

  """
  Flexible mol that tolerate mistakes in the smile
  """
  if sanitize:
    return MolFromSmiles(x)
  else:
    mol = MolFromSmiles(x, sanitize=False)
    GetSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def SmilesToDescriptors(fpgen, smile_list, sanitize=True):
    """Convert a list of SMILES to a DataFrame with fingerprints."""
    fingerprints = [list(fpgen.GetFingerprint(getMolFromSmile(x, sanitize=sanitize))) for x in smile_list]
    df = pd.DataFrame(fingerprints, index=smile_list)
    df = df_drop_single_value_columns(df)
    return df