from ord_schema.message_helpers import load_message, write_message
from ord_schema.proto import dataset_pb2
import pandas as pd

dataset = load_message("ord_dataset-c5b00523487a4211a194160edf45e9ab.pb.gz", dataset_pb2.Dataset)

def extract_reaction_data(reaction):
    """
    Extracts key reaction data, including undesired product yields.

    Returns:
    - catalyst: list of SMILES
    - reactants: list of SMILES
    - products: list of SMILES (desired)
    - products_undesired: list of SMILES (undesired)
    - solvent: list of SMILES
    - ee: list of floats for desired products
    - ee_undesired: list of floats for undesired products
    - yield_undesired_1: list with one float (first undesired yield)
    - yield_undesired_2: list with one float (second undesired yield)
    - reaction_smile: string with the reaction SMILES
    """

    catalyst = []
    reactants = []
    products = []
    products_undesired = []
    solvent = []

    ee = []
    ee_undesired = []

    yield_undesired = []

    reaction_smile = reaction.identifiers[0].value

    # Parse inputs to extract reactants, catalysts, solvents
    for key, val in reaction.inputs.items():
        for component in val.components:
            smile = component.identifiers[0].value
            role = component.reaction_role  # 1: reactant, 3: solvent, 4: catalyst
            if role == 1:
                reactants.append(smile)
            if role == 4:
                catalyst.append(smile)
            if role == 3 and smile not in solvent:
                solvent.append(smile)

    # Parse products to classify as desired or undesired
    for product in reaction.outcomes[0].products:
        if product.is_desired_product:
            products.append(product.identifiers[0].value)
            ee.append(product.measurements[0].percentage.value)
        else:
            products_undesired.append(product.identifiers[0].value)
            ee_undesired.append(product.measurements[0].percentage.value)

            # Extract relative yield (float_value), regardless of type
            yield_undesired.append(product.measurements[0].float_value.value)

    return (
        catalyst,
        reactants,
        products,
        products_undesired,
        solvent,
        ee,
        ee_undesired,
        yield_undesired,
        reaction_smile,
    )


df_smiles = pd.DataFrame(columns=['reactant_1','reactant_2', 'reactant_3', 'product_R', 'product_S', 'solvent', 'catalyst_1', 'catalyst_2',
 'product_undesired_R', 'product_undesired_S', 'ee_R', 'ee_S','ee_undesired_R','ee_undesired_S', 'yield_undesired_R','yield_undesired_S', 'reaction'])

for reaction in dataset.reactions:
  catalyst, reactants, products, products_undesired, solvent, ee, ee_undesired, yield_undesired, reaction_smile = extract_reaction_data(reaction)
  #print(len(reactants), len(products), len(solvent), len(products_undesired), len(ee), len(ee_undesired))
  df_smiles.loc[len(df_smiles.index)] = [*reactants, *products, *solvent, *catalyst, *products_undesired, *ee, *ee_undesired, *yield_undesired, reaction_smile]
import re

df_smiles['curated_catalyst_2'] = df_smiles['catalyst_2'].apply(lambda x: re.sub(re.escape('F[P](F)(F)(F)(F)F.CC(C)(C)C1=CC=[N@H]2C(=C1)C3=CC(=CC=[N@@H]3[Ir]2456c7cc(F)cc(F)c7C8=CC=C(C=[N]48)C(F)(F)F)C(C)(C)C.Fc9cc(F)c(C%10=[N]5C=C(C=C%10)C(F)(F)F)c6c9'),"F[P-](F)(F)(F)(F)F.CC(C)(C)C1=CC[N@H+]2C(=C1)C3=CC(=CC[N@@H+]3[Ir-4]2456c7cc(F)cc(F)c7C8=CC=C(C=[N+]48)C(F)(F)F)C(C)(C)C.Fc9cc(F)c(C%10=[N+]5C=C(C=C%10)C(F)(F)F)c6c9", x))
df_smiles.to_csv('dataset.csv', index=False)