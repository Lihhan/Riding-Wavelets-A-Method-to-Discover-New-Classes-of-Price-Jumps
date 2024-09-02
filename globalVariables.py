import numpy as np
import math

start_time = '2024-01-04'
end_time = '2024-01-04'

K = 60
S_K = (2 * math.log(K))**(-0.5)
C_K = (2 * math.log(K))**0.5 - (math.log(math.pi) + math.log(math.log(K))) / (2 * (2 * math.log(K))**0.5)
threshold = C_K - S_K * math.log(math.log(1/0.99))
omega_0 = 5 # Central frequency
anomaly = 20
j_values = [1, 2, 3, 4, 5, 6]
T = 119
num_components = 3
topRatio = 0.1
D1_window = 200
lower_ratio = 0.4
upper_ratio = 0.6
