from rdkit.Chem import GetSSSR, MolFromSmiles
from rdkit.Chem.AllChem import GetMorganGenerator
from baybe.utils.dataframe import df_drop_single_value_columns
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def getMolFromSmile(x, sanitize=True):

  """
  Get mol from SMILE with option to tolerate some mistakes in the SMILE.

  x:String, SMILE to convert to mol
  sanitize: Boolean, allow some mistakes in the smile if set to False

  return: Mol object
  """
  if sanitize:
    return MolFromSmiles(x)
  else:
    mol = MolFromSmiles(x, sanitize=False)
    GetSSSR(mol)
    mol.UpdatePropertyCache(strict=False)
    return mol


def SmilesToDescriptors(smile_list, method, sanitize=True, radius=2, fpSize=1024):
    """Convert a list of SMILES to a DataFrame with fingerprints.
    
    method: String, 'Morgan', 'Mordred' or 'Morfeus'
    smile_list: List of SMILES strings
    sanitize: Boolean, allow some mistakes in the smile if set to False

    return: Dataframe with the SMILE as index, the whole row is the fingerprint (each column is a bit).
    """
    mol_list = [getMolFromSmile(x, sanitize=sanitize) for x in smile_list]
    
    if method=='Morgan':
      fpgen = GetMorganGenerator(radius=radius,fpSize=fpSize)
      fingerprints = [list(fpgen.GetFingerprint(x)) for x in mol_list]
      df = pd.DataFrame(fingerprints, index=smile_list)
    
    if method=='Mordred':
      pass

    if method=='Morfeus':
      pass

    df = df_drop_single_value_columns(df)
    return df


def plot_results(results, lookup, figure_name, nbr_controls=1):
  """
  Plot the results from a BayBe simulation (observations, best observations, top 99% hits and cum. regret).

  results: dataframe from the simulation
  lookup: dataframe used for the simulation
  figure_name: name of the figure to save + extension (e.g 'figure.png')
  nbr_controls: number of control campaigns
  """
  if nbr_controls < 1:
    raise ValueError('You need at least one control campaign')

  nbr_campaing = results['Scenario'].nunique()
  dashes = [(1,0)]*(nbr_campaing-nbr_controls+1) + [(3,4)]*(nbr_controls-1)

  colors = sns.color_palette('Set1', nbr_campaing-nbr_controls)
  greys = sns.color_palette('Greys', nbr_controls)[::-1]
  palette = colors + greys

  optimum = lookup['ee_R'].max()
  #add columns
  results['ins_regret'] = optimum - results['ee_R_IterBest']
  results['sim_regret'] = optimum - results['ee_R_CumBest']
  results['cum_regret'] = results.groupby(['Scenario', 'Monte_Carlo_Run'], group_keys=False)['ins_regret'].cumsum()


  #Campaigns results
  iterMax = results['Iteration'].max()


  fig, ax = plt.subplots(2, 2, figsize=(10, 7))
  ax=ax.flatten()

 
  ax[0].hlines(y=lookup['ee_R'].max(), color='black', alpha=0.7, xmin=0, xmax=iterMax)
  sns.lineplot(data=results, x='Iteration', y='ee_R_IterBest', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[0])
  ax[0].set_ylabel('ee_R')
  ax[0].legend(fontsize=8)
  ax[0].set_title('Campaign results')


  #Cum best hit
  colors = sns.color_palette('Set1', 2)
  ax[1].hlines(y=lookup['ee_R'].max(), color='black', alpha=0.7, xmin=0, xmax=iterMax)
  sns.lineplot(data=results, x='Iteration', y='ee_R_CumBest', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[1])
  ax[1].set_title('Best hit')
  ax[1].legend(fontsize=8)
  ax[1].set_ylabel('ee_R')

  #top 99%
  ee_99 = lookup['ee_R'].quantile(0.99)
  results['top_hits'] = results.groupby(['Scenario', 'Monte_Carlo_Run'], group_keys=False)['ee_R_IterBest'] \
      .apply(lambda x: (x > ee_99).cumsum())

  sns.lineplot(data=results, x='Iteration', y='top_hits', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[2])
  ax[2].set_title('Top 99% hits')
  ax[2].set_ylabel('Number of top hits')
  ax[2].legend(fontsize=8)

  """
  sns.lineplot(data=results, x='Iteration', y='ins_regret', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[3])
  ax[3].set_title('Instantaneous regret')
  ax[3].set_ylabel('Ins. regret')
  ax[3].legend(fontsize=8)

  sns.lineplot(data=results, x='Iteration', y='sim_regret', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[4])
  ax[4].set_title('Simple regret')
  ax[4].set_ylabel('Simple regret')
  ax[4].legend(fontsize=8)
  """

  sns.lineplot(data=results, x='Iteration', y='cum_regret', hue='Scenario', style='Scenario', dashes=dashes, palette=palette, ax=ax[3])
  ax[3].set_title('Cumulative regret')
  ax[3].set_ylabel('Cum regret')
  ax[3].legend(fontsize=8)
  plt.tight_layout()
  plt.savefig('./figures/'+figure_name, dpi=300)