import numpy as np
from scipy.spatial.distance import cdist
from Loaders.LoadDataSeq import LoadDataSequence
from sklearn.preprocessing import LabelEncoder


class Seq2PrimeFree:
    """
    This class is used to convert a sequence free of priming affect.
    For a given dataset, one can clean data sequences of priming
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_loader = LoadDataSequence(
            dataset=dataset
        )
        self.df_prime_free = None

    def _get_first_trial(self, df, labels, col_name):
        """Get the first trial of the sequence."""
        first_trial = 0
        n_trial = int(df["trial"].max())
        cond = False
        while not cond and first_trial <= n_trial:
            first_trial += 1
            aux_labels = list(df[df["trial"] <= first_trial][col_name].unique())
            cond = True if aux_labels == labels else False
        return first_trial + 1


    def _remove_priming_samples(self, df):
        df_prime_free = df.copy()

        # Encode the labels for the sequences with more than three classes
        # label_enc = LabelEncoder()
        # y_orig = df_prime_free["y"].to_numpy()
        # y_enc = label_enc.fit_transform(y_orig)
        # df_prime_free["y_enc"] = y_enc
        if self.dataset == "seed7":
            col_label_name = "y_enc"
        else:
            col_label_name = "y"

        # Get the first trial from which the priming score will be calculated
        labels = list(df_prime_free[col_label_name].unique())
        init_trial = self._get_first_trial(df_prime_free, labels, col_label_name)

        df_prime_free["primed"] = np.nan
        df_prime_free["score"] = np.nan
        df_prime_free["assigned_class"] = np.nan

        n_trials = int(df["trial"].max())
        for trial in range(init_trial, n_trials + 1):
            # Take df of the trial
            df_trial = df_prime_free[df_prime_free["trial"] == trial]
            trial_data = np.stack(df_trial["x_norm"])
            trial_labels = df_trial[col_label_name].to_numpy()
            trial_label = int(trial_labels[0])

            prev_lab = df_prime_free[df_prime_free["trial"] == trial - 1][col_label_name].unique()[0]
            aux_prev_lab = np.ones_like(trial_labels) * prev_lab

            df_past = df_prime_free[df_prime_free["trial"] < trial]
            # Calculate centroids for each class
            past_data = np.stack(df_past["x_norm"])
            past_labels = np.array(df_past[col_label_name])
            unique_classes = np.unique(past_labels)
            centroids = {cls: np.mean(past_data[past_labels == cls], axis=0)
                         for cls in unique_classes}

            trial_to_centroid_distances = cdist(
                trial_data,
                np.array(list(centroids.values())),
                metric='cosine'
            )

            # Select the class that is closest to the current trial's samples
            trial_pred_clusters = np.argmin(trial_to_centroid_distances, axis=1)

            # Check if the predicted class is the same as the previous trial
            if prev_lab == trial_labels[0]:
                primed_samples = np.zeros_like(trial_pred_clusters)
                score_trial = np.zeros_like(trial_pred_clusters)
            else:
                primed_samples = (aux_prev_lab == trial_pred_clusters).astype(int)

                # Compute the priming score
                d_i_class = trial_to_centroid_distances[:, trial_label]
                d_i_prev = trial_to_centroid_distances[:, int(prev_lab)]
                d_i_other = trial_to_centroid_distances[:, ~np.isin(np.arange(len(unique_classes)), [trial_label, int(prev_lab)])]

                # Softmax approach
                thau = 0.1 # Makes the softmax more sensitive to the differences
                exp_class = np.exp(-d_i_class/thau)
                exp_prev = np.exp(-d_i_prev/thau)
                exp_other = np.exp(-d_i_other/thau)
                sum_exp = exp_class + exp_prev + np.sum(exp_other, axis=1)
                score_trial = exp_prev / sum_exp

            df_prime_free.loc[df_prime_free["trial"] == trial, "primed"] = primed_samples
            # trial_pred_clusters_dec = label_enc.inverse_transform(trial_pred_clusters)
            # df_prime_free.loc[df_prime_free["trial"] == trial, "assigned_class"] = trial_pred_clusters_dec
            df_prime_free.loc[df_prime_free["trial"] == trial, "assigned_class"] = trial_pred_clusters
            df_prime_free.loc[df_prime_free["trial"] == trial, "score"] = score_trial
            self.df_prime_free = df_prime_free


    def __call__(self, *args, **kwargs):
        pt = kwargs['pt']
        sess = kwargs['sess']

        df = self.seq_loader(pt=pt, sess=sess)
        self._remove_priming_samples(df)

        self.df_prime_free['score'] = self.df_prime_free['score'].fillna(0)
        self.df_prime_free['primed'] = self.df_prime_free['primed'].fillna(0)
        self.df_prime_free['assigned_class'] = (
            self.df_prime_free['assigned_class'].fillna(-1))


        return self.df_prime_free
