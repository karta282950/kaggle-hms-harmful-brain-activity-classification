import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torchtoolbox.tools import mixup_data, mixup_criterion
import sys, os
sys.path.append(os.path.join(sys.path[0],'bar','sub','dir'))
from Kaggle_KL_Div.kaggle_kl_div import score

from omegaconf import DictConfig
import hydra
import pandas as pd
import gc

'''
To Do:
* 

Done:
* 

Ref:
* https://www.researchgate.net/publication/359051366_GWNET_Detecting_Gravitational_Waves_using_Hierarchical_and_Residual_Learning_based_1D_CNNs
'''
    

class ResNet_1D_Block(nn.Module):
    def __init__(
            self, 
            cfg: DictConfig, 
            in_channels, 
            out_channels, 
            kernel_size,
            stride,
            padding,
            downsampling):
        super(ResNet_1D_Block, self).__init__()
        self.cfg = cfg
        self.bn1 = nn.BatchNorm1d(num_features=in_channels)
        self.relu = nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.bn2 = nn.BatchNorm1d(num_features=out_channels)
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        self.mp = nn.MaxPool1d(
            kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling
    def forward(self, x):
        identity = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out = self.mp(out)
        identity = self.downsampling(x)

        out+=identity
        return out

class EEGNet(nn.Module):
    def __init__(
            self,
            cfg: DictConfig,
            kernels,
            in_channels=20,
            fixed_kernel_size=17,
            num_classes=6):
        super(EEGNet, self).__init__()
        self.cfg = cfg
        self.kernels = kernels
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        
        for _, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                bias=False,)
            self.parallel_conv.append(sep_conv)

        self.bn1 = nn.BatchNorm1d(num_features=self.planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv1 = nn.Conv1d(
            in_channels=self.planes,
            out_channels=self.planes,
            kernel_size=fixed_kernel_size,
            stride=2,
            padding=2,
            bias=False,)
        self.block = self._make_resnet_layer(
            kernel_size=fixed_kernel_size,
            stride=1,
            padding=fixed_kernel_size//2)
        self.bn2 = nn.BatchNorm1d(num_features=self.planes)
        self.avgpool = nn.AvgPool1d(
            kernel_size=6,
            stride=6,
            padding=2)
        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True)
        self.fc = nn.Linear(
            in_features=424, out_features=num_classes)
        self.rnn1 = nn.GRU(
            input_size=156,
            hidden_size=156,
            num_layers=1,
            bidirectional=True)

    def _make_resnet_layer(
            self,
            kernel_size,
            stride,
            blocks=9,
            padding=0):
        layers = []
        downsample = None
        base_width = self.planes

        for i in range(blocks):
            downsampling = nn.Sequential(
                    nn.MaxPool1d(
                        kernel_size=2,
                        stride=2,
                        padding=0))
            layers.append(
                ResNet_1D_Block(
                    self.cfg, 
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling))

        return nn.Sequential(*layers)

    def forward(self, x):
        out_sep = []

        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv1(out)  

        out = self.block(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.avgpool(out)  
        
        out = out.reshape(out.shape[0], -1)  

        rnn_out, _ = self.rnn(x.permute(0,2, 1))
        new_rnn_h = rnn_out[:, -1, :]  

        new_out = torch.cat([out, new_rnn_h], dim=1)  
        result = self.fc(new_out)

        return result

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
        self.validation_step_outputs = []
        self.lin = nn.Softmax(dim=1)
        self.best_score = 1000.0
    def forward(self,images):
        logits = self.backbone(images)
        # logits = self.lin(logits)
        return logits
        
    def configure_optimizers(self):
        return get_optimizer(lr=self.cfg.LEARNING_RATE, params=self.parameters())

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
            #print(loss)
            print(y_pred)
            print(target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target = batch 
        y_pred = self(image)
        val_loss = self.loss_function(y_pred, target)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.validation_step_outputs.append({
            "val_loss": val_loss, "logits": y_pred, "targets": target})

        return {"val_loss": val_loss, "logits": y_pred, "targets": target}
    
    def train_dataloader(self):
        return self._train_dataloader
    
    def validation_dataloader(self):
        return self._validation_dataloader
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        output_val = nn.Softmax(dim=1)(torch.cat([x['logits'] for x in outputs],dim=0)).cpu().detach().numpy()
        target_val = torch.cat([x['targets'] for x in outputs],dim=0).cpu().detach().numpy()
        self.validation_step_outputs = []
        TARGETS=['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
        val_df = pd.DataFrame(target_val, columns = list(TARGETS))
        pred_df = pd.DataFrame(output_val, columns = list(TARGETS))

        val_df['id'] = [f'id_{i}' for i in range(len(val_df))] 
        pred_df['id'] = [f'id_{i}' for i in range(len(pred_df))] 

        print('val_df', val_df)
        val_df.to_csv('val_df.csv', sep=',')
        print('pred_df', pred_df)
        pred_df.to_csv('pred_df.csv', sep=',')
        avg_score = score(val_df, pred_df, row_id_column_name = 'id')

        if avg_score < self.best_score:
            print(f'Fold {self.fold}: Epoch {self.current_epoch} validation loss {avg_loss}')
            print(f'Fold {self.fold}: Epoch {self.current_epoch} validation KDL score {avg_score}')
            self.best_score = avg_score
            # val_df.to_csv(f'{Config.output_dir}/val_df_f{self.fold}.csv',index=False)
            # pred_df.to_csv(f'{Config.output_dir}/pred_df_f{self.fold}.csv',index=False)
        
        return {'val_loss': avg_loss,'val_cmap':avg_score}
    

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main(cfg):
    inputs = torch.ones((2, 8, 10000)) #raw eeg shape: (10000, 8)
    labels = torch.ones((2,6))
    model = EEGNet(cfg, kernels=[3,5,7,9], in_channels=8, fixed_kernel_size=5, num_classes=6)
    outputs = model(inputs)
    kl_loss = nn.KLDivLoss(reduction='batchmean')
    loss = kl_loss(outputs, labels)
    print(loss)

@hydra.main(config_path="./", config_name="config", version_base="1.1")
def main2(cfg):
    from datamodule import SegDataModule1D
    model = EEGNet(cfg, kernels=[3,5,7,9], in_channels=8, fixed_kernel_size=5, num_classes=6)
    datamodule = SegDataModule1D(cfg)
    datamodule.setup(stage=None)
    for inputs, labels in datamodule.train_dataloader():
        outputs = model(inputs)
        print(labels)
        print(outputs)
        break

if __name__ == '__main__':
    main2()