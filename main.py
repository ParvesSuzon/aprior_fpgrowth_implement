import pandas as pd
import time
import matplotlib.pyplot as plt
import psutil
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
import pyfpgrowth
import numpy as np

# Function to measure execution time
def measure_execution_time(func, *args):
    start_time = time.time()
    result = func(*args)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


# Function to measure memory usage
def measure_memory_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    return memory_usage


# Function to load and preprocess the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path, on_bad_lines='skip')
    dataset = df.values.tolist()
    dataset = [[str(item) for item in transaction] for transaction in dataset]
    return dataset


# Function to apply Apriori algorithm
def run_apriori(dataset, min_support):
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets


# Function to apply FP-Growth algorithm
def run_fpgrowth(dataset, min_support):
    patterns = pyfpgrowth.find_frequent_patterns(dataset, min_support)
    return patterns


# Dataset 1: Grocery Store Transactions
dataset1 = load_dataset('groceries.csv')

# Dataset 2: Online Retail Transactions
dataset2 = load_dataset('onlinestore.csv')

# Set minimum support for frequent itemsets
min_support = 0.01

# Set threshold time in seconds
threshold_time = 0.2

# Lists to store execution times and memory usage
execution_times = []
memory_usage = []

# Run Apriori algorithm on Dataset 1
apriori_result1, apriori_time1 = measure_execution_time(run_apriori, dataset1, min_support)
execution_times.append(apriori_time1)
memory_usage.append(measure_memory_usage())

# Run Apriori algorithm on Dataset 2
apriori_result2, apriori_time2 = measure_execution_time(run_apriori, dataset2, min_support)
execution_times.append(apriori_time2)
memory_usage.append(measure_memory_usage())

# Run FP-Growth algorithm on Dataset 1
fpgrowth_result1, fpgrowth_time1 = measure_execution_time(run_fpgrowth, dataset1, int(min_support * len(dataset1)))
execution_times.append(fpgrowth_time1)
memory_usage.append(measure_memory_usage())

# Run FP-Growth algorithm on Dataset 2
fpgrowth_result2, fpgrowth_time2 = measure_execution_time(run_fpgrowth, dataset2, int(min_support * len(dataset2)))
execution_times.append(fpgrowth_time2)
memory_usage.append(measure_memory_usage())

# Print results and execution times
print("Dataset Store - Apriori frequent itemsets:")
print(apriori_result1)
print("Execution time: {} seconds".format(apriori_time1))
print()

print("Dataset Grocery - Apriori frequent itemsets:")
print(apriori_result2)
print("Execution time: {} seconds".format(apriori_time2))
print()

print("Dataset Store - FP-Growth frequent itemsets:")
print(fpgrowth_result1)
print("Execution time: {} seconds".format(fpgrowth_time1))
print()

print("Dataset Grocery - FP-Growth frequent itemsets:")
print(fpgrowth_result2)
print("Execution time: {} seconds".format(fpgrowth_time2))
print()

# performance graph
algorithms = ["ApriGro", "ApriStor", "FGDGro", "FGDStor"]

# Generate time taking graph
x_pos = np.arange(len(algorithms))
colors = ['blue', 'red', 'green', 'orange']

plt.bar(x_pos, execution_times, color=colors)
plt.xlabel("Algorithm and Dataset")
plt.ylabel("Execution Time (seconds)")
plt.title("Algorithm Execution Time")
plt.xticks(x_pos, algorithms, rotation=45)
plt.axhline(y=threshold_time, color='gray', linestyle='--', linewidth=2, label = 'Threshold Time')
plt.legend()
plt.show()

# Generate memory utilization graph
plt.bar(x_pos, memory_usage, color=colors)
plt.xlabel("Algorithm and Dataset")
plt.ylabel("Memory Usage (MB)")
plt.title("Memory Utilization")
plt.xticks(x_pos, algorithms, rotation=45)
plt.show()
