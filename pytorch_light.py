import torch
from rdkit import Chem
import lightning as L
from pyli_utils import mols_from_file, rdkit2pyg, create_train_val_dataloaders, LightModule
from lightning.pytorch.callbacks import ModelCheckpoint


#### hyperparams
bs = 64
lr = 3e-4
max_epochs = 650
MEMORIZATION = True

#### datasets #! todo use whole data and valid
guacm_smiles = "/home/nobilm@usi.ch/master_thesis/guacamol/testdata.smiles"  
train_guac_mols = mols_from_file(guacm_smiles, True)

#### Memorization
if MEMORIZATION:
    obs = train_guac_mols[0]
    print(Chem.MolToSmiles(obs))
    train_data = rdkit2pyg([obs])
else:
    train_data = rdkit2pyg(train_guac_mols[:bs*4]) #! todo use whole data

#### dataloader  
train_dataset_loader, val_dataset_loader = create_train_val_dataloaders(train_data, train_data, bs, num_workers=0)


#### PyT PyL models
n_batches = len(train_dataset_loader)

cbs = [
    ModelCheckpoint(save_top_k=1, mode='min', monitor="total_val_loss", save_last=True),
]

graphRNN_light_model = LightModule(lr, n_batches, max_epochs)
trainer = L.Trainer(
    callbacks=cbs,
    max_epochs=max_epochs,
    accelerator="gpu",
    devices=1,
    # precision="16-mixed" #"bf16-mixed"
    log_every_n_steps=500
)

# from lightning.pytorch.tuner import Tuner
# tuner = Tuner(trainer)
# lr_finder = tuner.lr_find(graphRNN_light_model, train_dataset_loader) # '0.000016'

try:
    #### Train
    trainer.fit(
        model=graphRNN_light_model,
        train_dataloaders=train_dataset_loader,
        # val_dataloaders=val_loader
    )
except KeyboardInterrupt:     
    graphRNN_light_model.generate_mols(10)

graphRNN_light_model.generate_mols(10)

#  edge_loss=3.85e-7,