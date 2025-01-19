import numpy as np
import matplotlib.pyplot as plt


ratios = [0.1, 0.2, 0.3, 0.4, 0.5]


norm = [13.70, 21.71, 22.19, 26.21, 22.77]
bound = [12.60, 15.17, 16.49, 21.92, 22.33]
test_acc = [96.33, 94.73, 94.65, 94.37, 93.62]  
test_f1 = [54.92, 81.6, 87.31, 89.56, 90.23]


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
