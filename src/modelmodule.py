from torch import nn 
import torch 

class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self):
        super().__init__(reduction="batchmean")

    def forward(self, y, t):
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t)
        return loss

def get_optimizer(lr, params):
    model_optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, params), 
            lr=lr,
            weight_decay=1e-2)
    interval = "epoch"
    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer, 
        T_0=20, 
        T_mult=1, 
        eta_min=1e-6, 
        last_epoch=-1)
    return {"optimizer": model_optimizer, 
            "lr_scheduler":{
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1}}

class EEGModel(pl.LightningModule):
    def __init__(self, cfg, num_classes = 6, pretrained = False, fold = 5):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.fold = fold
        self.backbone = EEGNet(cfg, kernels=[3,5,7,9], in_channels=8, fixed_kernel_size=5, num_classes=self.num_classes)
        self.loss_function = KLDivLossWithLogits() #nn.KLDivLoss() #nn.BCEWithLogitsLoss() 
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.lin = nn.Softmax(dim=1)
        self.best_score = 1000.0
    def forward(self,images):
        logits = self.backbone(images)
        # logits = self.lin(logits)
        return logits
        
    def configure_optimizers(self):
        return get_optimizer(lr=8e-3, params=self.parameters())

    def train_with_mixup(self, X, y):
        X, y_a, y_b, lam = mixup_data(X, y, alpha=0.1)
        y_pred = self(X)
        loss_mixup = mixup_criterion(KLDivLossWithLogits(), y_pred, y_a, y_b, lam)
        return loss_mixup

    def training_step(self, batch, batch_idx, use_mixup=False):
        image, target = batch   
        if use_mixup:
            loss = self.train_with_mixup(image, target)
        else:
            y_pred = self(image)
            loss = self.loss_function(y_pred, target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.training_step_outputs.append(torch.tensor([loss]))
        return loss
    
    def on_train_epoch_end(self):
        epoch_average = torch.stack(self.training_step_outputs).mean()
        self.log("training_epoch_average", epoch_average)
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        image, target = batch 
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.validation_step_outputs.append(torch.tensor([val_loss]))
        
        return val_loss
    
    def on_validation_epoch_end(self):
        epoch_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear() 

    def train_dataloader(self):
        return self._train_dataloader
    
    def validation_dataloader(self):
        return self._validation_dataloader
    
    
    

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    import warnings
    warnings.filterwarnings("ignore")
    inputs = torch.rand(2, 8, 10000) #raw eeg shape: (10000, 8)
    labels = torch.rand(2,6)
    model = EEGNet(cfg, kernels=[3,5,7,9], in_channels=8, fixed_kernel_size=5, num_classes=6)
    outputs = model(inputs)
    loss = KLDivLossWithLogits()(outputs, labels)
    print(loss)

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main2(cfg):
    from datamodule import SegDataModule1D
    import warnings
    warnings.filterwarnings("ignore")
    model = EEGNet(cfg, kernels=[3,5,7,9], in_channels=8, fixed_kernel_size=5, num_classes=6)
    datamodule = SegDataModule1D(cfg)
    datamodule.setup(stage=None)
    for inputs, labels in datamodule.train_dataloader():
        outputs = model(inputs)
        loss = KLDivLossWithLogits()(outputs, labels)
        print(loss)
        break

if __name__ == '__main__':
    main()