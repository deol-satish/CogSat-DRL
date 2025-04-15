import numpy as np

channelFreqs = 1e9 * np.array([1.498875, 1.500125, 1.500375, 1.500625, 1.500875,
                               1.501125, 1.501375, 1.501625, 1.501875, 1.502125])

channelFreqs = [1e9 * x for x in [1.498875, 1.500125, 1.500375, 1.500625, 1.500875,
                                  1.501125, 1.501375, 1.501625, 1.501875, 1.502125]]


default_dataset_path = "data/Satellite_Australia_Simulation_Log_2025_04_11_v2.csv"
debug_output = True  # Global flag to control print behavior


channelFreqs.append(0)

n_leo = 3  # Number of LEO satellites
n_geo = 1  # Number of GEO satellites
n_leo_users = 2  # Number of LEO users
n_geo_users = 2  # Number of GEO users