Files amiable at:
https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1

README for dataset ZINC_full

This dataset contains ZINC_train, ZINC_test and ZINC_val:
ZINC_train:	Graph 1		- 220011
ZINC_test:	Graph 220012	- 225011
ZINC_val:	Graph 225012	- 249456


=== Usage ===

This folder contains the following comma separated text files 
(replace DS by the name of the dataset):

n = total number of nodes
m = total number of edges
N = number of graphs


=== Description of the dataset === 

The node labels are atom types and the edge labels atom bond types.

Node labels:

'C': 0
'O': 1
'N': 2
'F': 3
'C H1': 4
'S': 5
'Cl': 6
'O -': 7
'N H1 +': 8
'Br': 9
'N H3 +': 10
'N H2 +': 11
'N +': 12
'N -': 13
'S -': 14
'I': 15
'P': 16
'O H1 +': 17
'N H1 -': 18
'O +': 19
'S +': 20
'P H1': 21
'P H2': 22
'C H2 -': 23
'P +': 24
'S H1 +': 25
'C H1 -': 26
'P H1 +': 27

Edge labels:

'SINGLE': 1
'DOUBLE': 2
'TRIPLE': 3

=== Source ===

https://ml4physicalsciences.github.io/files/NeurIPS_ML4PS_2019_93.pdf
@article{bresson2019two,
title={A Two-Step Graph Convolutional Decoder for Molecule Generation},
author={Bresson, Xavier and Laurent, Thomas},
journal={ Workshop on Machine Learning and the Physical Sciences (NeurIPS 2019),
Vancouver, Canada, arXiv preprint arXiv:1906.03412},
year={2019}
}

The chemical property used is y(m) = logP(m) − SA(m) − cycle(m) taken from
this paper, section 3.2:
https://arxiv.org/pdf/1802.04364.pdf
@article{jin2018junction,
title={Junction tree variational autoencoder for molecular graph generation},
author={Jin, Wengong and Barzilay, Regina and Jaakkola, Tommi},
journal={arXiv preprint arXiv:1802.04364},
year={2018}
}
