import os
import subprocess
import time
import torch.multiprocessing as mp
import csv
from multiprocessing import Lock
from training.train_clean_sequence import train_clean_sequence
from itertools import product
import argparse
from utils.constants import MODEL2SAVE
import torch
from Tools.Seq2PrimeFree import Seq2PrimeFree


def cleanup_memory():
    torch.cuda.empty_cache()

def get_free_memory():
    """Returns the free memory of the GPU in MB."""
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True)
    free_memory = int(int(result.stdout.strip().split("\n")[0]) * 0.8)
    return free_memory

def log_results(filename, seq_len, pt, sess, seed, acc, f1, pe, pe_norm, lock):
    """Logs experiment results into a shared CSV file."""
    with lock:
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([seq_len, pt, sess, seed, acc, f1, pe, pe_norm])


# def run_experiment(backbone, seq_len, pt, sess, seed, session, lock, filename):
def run_experiment(backbone, loader, pt, sess, seq_len, seed, lock, filename):
    """Runs an experiment on the GPU and logs results."""

    acc, f1, PE, PE_norm = train_clean_sequence(backbone, loader, pt, sess, seq_len, seed)
    cleanup_memory()
    log_results(filename, seq_len, pt, sess, seed, acc, f1, PE, PE_norm, lock)


def main(args):
    backbone = args.backbone
    dataset = args.dataset

    loader = Seq2PrimeFree(dataset)

    if dataset == ("seed" or "seed5"):
        participants = [i for i in range(1, 16)]
        sessions = [i for i in range(1, 4)]
    elif dataset == "seed7":
        participants = [i for i in range(1, 21)]
        sessions = [i for i in range(1, 5)]

    # seq_len = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    seq_len = [2]
    seeds = [42, 69, 23, 5, 12]
    # Define the pool of experiments to run
    experiments = list(product(seq_len, participants, sessions, seeds))

    active_processes = []
    model_memory = 1000  # Memory occupied by each model in MB
    lock = Lock()  # Lock for safe file writing

    # Initialize CSV file
    path = os.path.join(MODEL2SAVE,
                        f"clean-sequences/{backbone}/{dataset}")
    os.makedirs(path, exist_ok=True)
    results_path =os.path.join(path, f'{backbone}_results.csv')
    with open(results_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["seq_len", "pt", "sess", "seed", "acc", "f1", "pe", "pe_norm"])

    while experiments:
        free_memory = get_free_memory()
        available_slots = free_memory // model_memory
        while available_slots > 0 and experiments:
            seq_len, pt, sess, seed = experiments.pop(0)
            proc = mp.Process(target=run_experiment, args=(
                backbone, loader, pt, sess, seq_len, seed, lock, results_path))
            proc.start()
            active_processes.append(proc)
            available_slots -= 1

        # Clean up finished processes
        for p in active_processes[:]:  # Iterate over a copy
            if not p.is_alive():
                p.join()  # Ensure proper cleanup before removing from the list
                active_processes.remove(p)

        time.sleep(30)  # Check every 5 seconds for available memory

if __name__ == '__main__':
    mp.set_start_method("spawn")  # Ensure multiprocessing works properly

    parser = argparse.ArgumentParser(description='Run main')
    parser.add_argument('--backbone', type=str, default="transformer")
    parser.add_argument('--dataset', type=str, default="seed")
    args = parser.parse_args()
    main(args)