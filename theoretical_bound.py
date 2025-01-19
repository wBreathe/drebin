import os
import pickle
import numpy as np
import math
import csv
def error_bound(
        train_loss: float,
        m: int,
        delta: float,
        norm_diff: np.float64
        ):
    complexity = 2 * np.log((m+1)/delta)
    bound = math.sqrt((norm_diff**2 + complexity) / (m))
    return bound + train_loss


if __name__ == "__main__":
    
    with open("temp/num-100_kernel-linear_testSize-0.2_priorPortion-0.2_stats_1737312084", "rb") as f:
        record = pickle.load(f)
    total_samples = 34208 
    delta = 0.05
    bounds = [("norm_diff", "train_loss", "test_loss", "bound")]
    for _, v in record.items():
        assert("full" in v.keys() and "train_loss" in v.keys() and "test_loss" in v.keys())
        norm_diff = v["full"][0]
        train_loss = v["train_loss"][0]
        test_loss = v["test_loss"][0]
        bounds.append((norm_diff, train_loss, test_loss, error_bound(train_loss, total_samples, delta, norm_diff)))
    
    csv_file = "output_0.2.csv"

    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(bounds)