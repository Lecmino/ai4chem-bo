# ai4chem-bo

This repository reference all the work done for this project, assessing the performances and abilities of [BayBE](https://emdgroup.github.io/baybe/stable/index.html) to explore the chemical space of [HTE experiments found in the ORD database](https://open-reaction-database.org/dataset/ord_dataset-c5b00523487a4211a194160edf45e9ab), provided by the author of [this paper](https://doi.org/10.1038/s41467-023-42446-5). The following notebooks and scripts can be found:

[**process_database.py**](./process_database.py): Extract the reactions features (smiles, ee, yield) from the ord dataset, which was downloaded from the ORD database. Can be downloaded from the [ORD GitHub](https://github.com/open-reaction-database/ord-data/tree/main/data/c5), or using this [link](https://github.com/open-reaction-database/ord-data/raw/refs/heads/main/data/c5/ord_dataset-c5b00523487a4211a194160edf45e9ab.pb.gz?download=).

[**display_dataset.ipynb**](./display_database.ipynb): The notebook provides different basic statistics about the dataset. It also displays the different chemicals involved in the reactions.

[**bo_initialization.ipynb**](./bo_initialization.ipynb): This notebook provide the preliminary results obtained using the Morgan fingerprint, and evaluate the effect of different initial sampling sequences. In a second part, the notebook explore the different surrogate models available in _BayBE_.

[**bo_insights.ipynb**](./bo_insights.ipynb): Provide alternative vizualisation of the path taken by the optimizer in the reaction space.

[**bo_fingerprints.ipynb**](./bo_fingerprints_batch.ipynb): This notebook evaluate different fingerprints and their impact on the performance on the optimizer.

[**bo_multiopt.ipynb**](./bo_multiopt.ipynb): Here, the dual optimization abilities of _BayBE_ are assessed with a maximizing ee - minimizing amount of undesired product dual optimization procedure.
