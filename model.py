import timm 
from torch import nn 
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch.nn.functional as F
import hydra
class CustomModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig, num_classes: int = 6, pretrained: bool = True):
        super(CustomModel, self).__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.USE_KAGGLE_SPECTROGRAMS = True
        self.USE_EEG_SPECTROGRAMS = True
        self.model = timm.create_model(
            cfg.model,
            pretrained=pretrained,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        if cfg.FREEZE:
            for i,(name, param) in enumerate(list(self.model.named_parameters())\
                                             [0:cfg.NUM_FROZEN_LAYERS]):
                param.requires_grad = False

        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.custom_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, num_classes)
        )
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def __reshape_input(self, x):
        """
        Reshapes input (128, 256, 8) -> (512, 512, 3) monotone image.
        """ 
        # === Get spectograms ===
        spectograms = [x[:, :, :, i:i+1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=1)
        
        # === Get EEG spectograms ===
        eegs = [x[:, :, :, i:i+1] for i in range(4,8)]
        eegs = torch.cat(eegs, dim=1)
        
        # === Reshape (512,512,3) ===
        if self.USE_KAGGLE_SPECTROGRAMS & self.USE_EEG_SPECTROGRAMS:
            x = torch.cat([spectograms, eegs], dim=2)
        elif self.USE_EEG_SPECTROGRAMS:
            x = eegs
        else:
            x = spectograms
            
        x = torch.cat([x,x,x], dim=3)
        x = x.permute(0, 3, 1, 2)
        return x
    
    def forward(self, x):
        x = self.__reshape_input(x)
        x = self.features(x)
        x = self.custom_layers(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(out, y)
        self.log('train_loss', torch.tensor([loss]), prog_bar=True)
        self.training_step_outputs.append(torch.tensor([loss]))
        return loss
    
    def on_train_epoch_end(self):
        # 计算平均loss
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        out = F.log_softmax(out, dim=1)
        kl_loss = nn.KLDivLoss(reduction='batchmean')
        loss = kl_loss(out, y)
        self.log('val_loss', torch.tensor([loss]), prog_bar=True)
        self.validation_step_outputs.append(torch.tensor([loss]))
        return loss
    
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear() 

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return F.softmax(self(batch), dim=1)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.EPOCHS)
        return [optimizer], [scheduler]

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    model = CustomModel(cfg)
    inputs = torch.zeros((3, 128, 256, 8))
    y = torch.zeros((3, 6))
    outputs = model(inputs)
    outputs = F.log_softmax(outputs, dim=1)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    loss = kl_loss(outputs, y)
    print(loss)
if __name__ == '__main__':
    main()