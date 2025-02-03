import numpy as np
import matplotlib.pyplot as plt


ratios = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]


norm = [20.98, 25.15, 29.41, 29.43, 25.47]
bound = [12.27, 15.23, 20.05, 23.99, 24.65]
test_acc = [95.40, 94.98, 94.78, 94.34, 93.60]  
test_f1 = [77.37, 85.49, 88.79, 90.04, 90.23]


def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


norm_normalized = normalize(np.array(norm))
bound_normalized = normalize(np.array(bound))
test_acc_normalized = normalize(np.array(test_acc))
test_f1_normalized = normalize(np.array(test_f1))


plt.figure(figsize=(8, 6))
plt.plot(ratios, norm_normalized, label='Norm', marker='o')
plt.plot(ratios, bound_normalized, label='Bound', marker='s')
plt.plot(ratios, test_acc_normalized, label='Test Acc', marker='^')
plt.plot(ratios, test_f1_normalized, label='Test F1', marker='x')


plt.legend()
plt.title("Partition")
plt.xlabel("Ratios")
plt.ylabel("Normalized Values")
plt.grid()


plt.savefig("images/partition.pdf", format='pdf', bbox_inches='tight')
plt.show()
