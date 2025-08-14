import torch
import random
import numpy as np
from utils.constants import emotion2num
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed):
    # Set seed for Python random
    random.seed(seed)
    # Set seed for NumPy
    np.random.seed(seed)
    # Set seed for PyTorch
    torch.manual_seed(seed)
    # If using CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_dataloaders(x_train, y_train, x_val, y_val,  x_test, y_test, batch_size=256):
    train_ds = TensorDataset(
        torch.tensor(x_train).float(),
        torch.tensor(y_train)
    )

    val_ds = TensorDataset(
        torch.tensor(x_val).float(),
        torch.tensor(y_val)
    )


    test_ds = TensorDataset(
        torch.tensor(x_test).float(),
        torch.tensor(y_test)
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
        drop_last=True
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=True,
    )

    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        persistent_workers=True,
        drop_last=True
    )
    return train_dl, val_dl, test_dl



def compute_priming_error(model, dataset, df):
    # Set the model to evaluate mode
    model.eval()
    model = model.to("cpu")

    # Get testing trials
    if dataset == "seed":
        test_trials = [10, 11, 12, 13, 14, 15]
    elif dataset == "seed7":
        test_trials = [16, 17, 18, 19, 20]
    elif dataset == "seed5":
        test_trials = [11, 12, 13, 14, 15]

    n_misc = 0 # total number of samples misclassified
    n_pe = 0  #Total number of priming errors
    n_total_samples = 0 # Total number of samples in the test set

    for trial in test_trials:
        if not df["trial"].isin([trial]).any(): continue

        y_trial = int(df[df['trial'] == trial]['y'].unique()[0])

        # TODO: Fix this problem...
        try:
            y_prev_trial = int(df[df['trial'] == trial - 1]['y'].unique()[0])
        except (IndexError, KeyError):
            try:
                y_prev_trial = int(df[df['trial'] == trial - 2]['y'].unique()[0])
            except (IndexError, KeyError):
                try:
                    y_prev_trial = int(df[df['trial'] == trial - 3]['y'].unique()[0])
                except (IndexError, KeyError):
                    y_prev_trial = int(df[df['trial'] == trial - 4]['y'].unique()[0])

        # If the trial is the same as the previous one, skip. No chance of priming
        if y_trial == y_prev_trial: continue

        x_trial = np.stack(df[df['trial'] == trial]['x'])

        x_trial_tensor = torch.tensor(x_trial).float()
        _, y_preds = model(x_trial_tensor)
        y_preds = y_preds.argmax(dim=1).detach().numpy()

        # Calculate how many samples are misclassified by the previous trial
        y_prev_aux = np.ones(y_preds.shape[0]) * y_prev_trial
        n_pe_trial = np.sum(y_preds == y_prev_aux)

        # Calculate how many samples are misclassified in the current trial
        y_aux = np.ones(y_preds.shape[0]) * y_trial
        n_misc_trial = np.sum(y_preds != y_aux)

        # Add the number of priming errors and misclassified samples
        n_pe += n_pe_trial
        n_misc += n_misc_trial
        n_total_samples += y_preds.shape[0]

    pe = np.around(n_pe / n_total_samples, 3)
    pe_norm = np.around(n_pe / n_misc, 3)

    return pe, pe_norm

