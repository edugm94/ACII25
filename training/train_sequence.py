import os
import yaml
import numpy as np
import pandas as pd
import lightning as L
from utils.constants import *
from Models.LitModel import LitModel
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from Loaders.LoadDataSeq import LoadDataSequence
from Models.backbones import LSTM, Transformer, MLP
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import (LearningRateMonitor,
                                         ModelCheckpoint, EarlyStopping)
from sklearn.utils.class_weight import compute_class_weight
from utils.funcs import set_seed, build_dataloaders, compute_priming_error



def apply_sliding_window(data, seq_len, keep_last=False):
    sw_data = np.lib.stride_tricks.sliding_window_view(
        data, seq_len, axis=0
    )
    sw_data = sw_data.transpose(0, 2, 1) if sw_data.ndim == 3 else sw_data
    return sw_data[:, -1] if keep_last else sw_data



def train_sequence(model_name, loader, pt, sess, seq_len, seed):
    dataset = loader.dataset

    # Load config file for model training
    with open("learn_config.yaml", 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    # seed everything
    set_seed(seed)

    # Load data and prepare dataloaders
    df = loader(pt=pt, sess=sess)

    # Separate
    if dataset == "seed":
        df_train = df[df["trial"] < 10]
        df_test = df[df["trial"] >= 10]

    elif dataset == "seed5":
        df_train = df[df["trial"] < 11]
        df_test = df[df["trial"] >= 11]

    elif dataset == "seed7":
        df_train = df[df["trial"] < 16]
        df_test = df[df["trial"] >= 16]
    else:
        raise ValueError("Invalid dataset")

    x_train = np.stack(df_train["x"])
    x_test = np.stack(df_test["x"])
    trial_train = np.stack(df_train["trial"])
    trial_test = np.stack(df_test["trial"])

    if dataset == "seed7":
        y_train = np.stack(df_train["y_enc"])
        y_test = np.stack(df_test["y_enc"])
    else:
        y_train = np.stack(df_train["y"])
        y_test = np.stack(df_test["y"])

    # scale data
    x_train = scale(x_train)
    x_test = scale(x_test)

    if seq_len > 0:
        x_train = apply_sliding_window(x_train, seq_len)
        x_test = apply_sliding_window(x_test, seq_len)
        y_train = apply_sliding_window(y_train, seq_len, keep_last=True)
        y_test = apply_sliding_window(y_test, seq_len, keep_last=True)
        trial_train = apply_sliding_window(trial_train, seq_len, keep_last=True)
        trial_test = apply_sliding_window(trial_test, seq_len, keep_last=True)

        df_train = pd.DataFrame({
                "x": list(x_train),
                "y": y_train,
                "trial": trial_train
        })

        df_test = pd.DataFrame({
            "x": list(x_test),
            "y": y_test,
            "trial": trial_test
        })

    # obtain class weights
    labels = np.concatenate([y_train, y_test], axis=0)
    n_classes = len(np.unique(labels))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )

    # Build the validation set:
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        stratify=y_train,
        test_size=0.15,
        random_state=seed
    )

    # Create dataloaders
    train_dl, val_dl, test_dl = build_dataloaders(
        x_train, y_train,
        x_val, y_val,
        x_test, y_test,
        batch_size=cfg["bs"])

    #####################
    # Instantiate model
    if model_name == "lstm":
        backbone = LSTM(n_classes=n_classes)

    elif model_name == "transformer":
        backbone = Transformer(
            n_classes=n_classes,
        )
    elif model_name == "mlp":
        backbone = MLP(
            n_classes=n_classes
        )

    else:
        raise ValueError("Model not implemented")

    cfg["class_weights"] = class_weights
    cfg["n_classes"] = n_classes
    model = LitModel(
        backbone=backbone,
        config=cfg
    )

    #####################
    # Declare path to save model and prepare callbacks
    save_dir = os.path.join(MODEL2SAVE,
                            f"original-sequences/{model_name}/{dataset}/"
                            f"seq-len-{seq_len}/rand-s{seed}/PT{pt}/Sess{sess}/")

    # Create callbacks and logger
    logger = TensorBoardLogger(
        save_dir=save_dir,
        version="tensorboard",
        name="logs",
    )

    checkpoint_cb = ModelCheckpoint(
        dirpath=save_dir,
        enable_version_counter=False,
        filename='checkpoint',  # Filename for the saved checkpoint
        monitor='val_loss',  # Monitor the training loss (or any other metric)
        mode='min'
    )

    early_stop_cb = EarlyStopping(
        monitor='val_loss',  # Metric to monitor
        patience=5,
        verbose=False,  # Verbosity mode
        mode='min',
        min_delta=0.01
    )

    # Create trainer
    trainer = L.Trainer(
        default_root_dir=save_dir,
        max_epochs=500,
        devices=1,
        callbacks=[
            checkpoint_cb,
            early_stop_cb,
            LearningRateMonitor(logging_interval='epoch')
        ],
        logger=logger,
        log_every_n_steps=5
    )

    # Call trainer.fitz
    trainer.fit(model, train_dl, val_dl)

    # Call trainer.test
    metrics = trainer.test(model, test_dl)

    # Extract the metrics
    acc = np.around(metrics[0]["test_accuracy"], 3)
    f1 = np.around(metrics[0]["test_f1"], 2)

    dfs = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    pe, pe_norm = compute_priming_error(model, dataset, dfs)
    return acc, f1, pe, pe_norm








