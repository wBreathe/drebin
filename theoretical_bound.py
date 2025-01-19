import os
import pickle
import numpy as np
import math
import csv
import re
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
    samples={"0.1": 35277, "0.2": 34208, "0.3": 26326, "0.4":18225, "0.5": 13018}
    for file_name in os.listdir("./temp/"):
        with open(os.path.join("temp", file_name), "rb") as f:
            record = pickle.load(f)
        
        match = re.search(r'testSize-([\d\.]+)', file_name)
        if match:
            test_size = match.group(1)
        print("testSize:", test_size)
        total_samples = samples[test_size] 
        delta = 0.05
        bounds = [("norm_diff", "train_loss", "test_loss", "bound")]
        for _, v in record.items():
            assert("full" in v.keys() and "train_loss" in v.keys() and "test_loss" in v.keys())
            norm_diff = v["full"][0]
            train_loss = v["train_loss"][0]
            test_loss = v["test_loss"][0]
            bounds.append((norm_diff, train_loss, test_loss, error_bound(train_loss, total_samples, delta, norm_diff)))
    
        csv_file = f"{file_name}.csv"

        with open(os.path.join("results", csv_file), mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(bounds)