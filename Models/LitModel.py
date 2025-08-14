import numpy as np
import torch
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score


class LitModel(L.LightningModule):
    def __init__(self, backbone, config):
        super(LitModel, self).__init__()
        self.config = config
        # Backbone network to get logit representations
        self.backbone = backbone

        # Save model's hyperparameters
        self.save_hyperparameters(config)

        # Optimizer parameters
        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.class_weights = torch.from_numpy(
            np.array(config["class_weights"])
        )

        # lists to save step loss values
        self.loss_train_epoch = []
        self.loss_val_epoch = []

        # Metrics for assess the model
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=config["n_classes"]
        )
        self.train_f1 = MulticlassF1Score(
            num_classes=config["n_classes"],
            average="macro"
        )

        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=config["n_classes"]
        )
        self.val_f1 = MulticlassF1Score(
            num_classes=config["n_classes"],
            average="macro"
        )

        self.test_acc = Accuracy(
            task="multiclass",
            num_classes=config["n_classes"]
        )
        self.test_f1 = MulticlassF1Score(
            num_classes=config["n_classes"],
            average="macro"
        )

    def forward(self, inputs):
        # feats, logits = self.backbone(inputs)
        # return feats, logits
        feats, logits = self.backbone(inputs)
        return feats, logits

    def training_step(self, batch, batch_idx):
        x, targets = batch
        feats, logits = self(x)
        # logits = self(x)

        # Get class weight for samples in the batch
        class_weights = self.class_weights.to(self.device).float()

        preds = F.log_softmax(logits, dim=1)
        targets = targets.long()
        loss = F.nll_loss(preds, targets, weight=class_weights)

        self.loss_train_epoch.append(loss.item())
        self.train_acc(preds, targets)
        self.train_f1(preds, targets)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        feats, logits = self(x)
        # logits = self(x)

        # Get class weight for samples in the batch
        class_weights = self.class_weights.to(self.device).float()

        preds = F.log_softmax(logits, dim=1)
        targets = targets.long()
        loss = F.nll_loss(preds, targets, weight=class_weights)

        self.loss_val_epoch.append(loss.item())
        self.val_acc(preds, targets)
        self.val_f1(preds, targets)
        return loss


    def test_step(self, batch, batch_idx):
        x, targets = batch
        feats, logits = self(x)
        # logits = self(x)

        preds = F.softmax(logits, dim=1)
        preds = torch.argmax(preds, dim=1)
        self.test_acc(preds, targets)
        self.test_f1(preds, targets)


    def on_train_epoch_end(self) -> None:
        avg_loss = torch.tensor(self.loss_train_epoch).mean()
        train_acc = self.train_acc.compute()
        train_f1 = self.train_f1.compute()

        self.logger.experiment.add_scalars("Metric/Loss", {"Train": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Metric/Acc", {"Train": train_acc}, self.current_epoch)
        self.logger.experiment.add_scalars("Metric/F1", {"Train": train_f1}, self.current_epoch)


        self.log("train_loss", avg_loss, prog_bar=True)
        self.loss_train_epoch.clear()
        self.train_acc.reset()
        self.train_f1.reset()

    def on_validation_epoch_end(self) -> None:
        avg_loss = torch.tensor(self.loss_val_epoch).mean()
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()

        self.logger.experiment.add_scalars("Metric/Loss", {"Val": avg_loss}, self.current_epoch)
        self.logger.experiment.add_scalars("Metric/Acc", {"Val": val_acc}, self.current_epoch)
        self.logger.experiment.add_scalars("Metric/F1", {"Val": val_f1}, self.current_epoch)

        self.log("val_loss", avg_loss, prog_bar=True)
        self.loss_val_epoch.clear()
        self.val_acc.reset()
        self.val_f1.reset()

    def on_test_epoch_end(self):
        test_acc = self.test_acc.compute()
        test_f1 = self.test_f1.compute()

        # Log or return metrics here before resetting
        self.log('test_accuracy', test_acc, prog_bar=True)
        self.log('test_f1', test_f1, prog_bar=True)

        # # Reset the metrics
        self.test_acc.reset()
        self.test_f1.reset()

        # Return computed values for consistency
        return {"accuracy": test_acc.item(), "f1": test_f1.item()}


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=float(self.config["lr"]),
                                     weight_decay=float(self.config["weight_decay"]))

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, min_lr=1e-7, factor=0.8)
        return [optimizer], [dict(scheduler=lr_scheduler, interval="epoch",
                                  monitor="val_loss")]
