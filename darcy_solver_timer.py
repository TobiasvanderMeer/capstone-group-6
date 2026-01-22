import time
import numpy as np

import utils

# this file times the performance of the provided darcy solver

n = 64
MAX_SAMPLES = 50

print("loading data")
true_test_x = utils.load_x_true_test().reshape((-1, 3600))

true_test_x64 = utils.load_x_true_test64().reshape((-1, 4096))

if n == 60:
    x_set = true_test_x
else:
    x_set = true_test_x64

n_samples = min(x_set.shape[0], MAX_SAMPLES)

prediction = np.empty(x_set.shape)

setup_time = time.time()
from jeffrey_code import solve_darcy_flow, source_function
f = source_function(n)
start_time = time.time()

for j in range(n_samples):
    prediction[j] = solve_darcy_flow(n, x_set[j], f)
    if j == 0:
        first_batch_time = time.time()

end_time = time.time()

print(f"total time: {end_time-setup_time:.4f}s, "
          f"avg time per sample: {(end_time - start_time)/n_samples*1000:.2f}ms, "
          f"setup: {start_time - setup_time:.3f}s, "
          f"first batch {(first_batch_time-start_time)*1000}ms\n")






