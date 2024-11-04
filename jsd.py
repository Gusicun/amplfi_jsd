import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from pathlib import Path


file_path_1 = "/home/fan.zhang/bilby/samples_bilby2.csv"
file_path_2 = "/home/fan.zhang/base/pl-logdir/phenomd-60-transforms-4-4-resnet-wider-dl/amplfi_samples_5000.csv"


data1 = pd.read_csv(file_path_1)
data2 = pd.read_csv(file_path_2)


if 'phi_12' in data1.columns and 'phi_jl' in data1.columns:
    data1['phi'] = (data1['phi_12'] + data1['phi_jl']) / 2


parameters = ["chirp_mass", "mass_ratio", "luminosity_distance", "phase", "theta_jn", "dec", "psi", "phi"]

def calculate_jsd(p, q, num_bins=100):
    p_hist, bin_edges = np.histogram(p, bins=num_bins, density=True)
    q_hist, _ = np.histogram(q, bins=bin_edges, density=True)
    p_hist += 1e-10  
    q_hist += 1e-10
    return jensenshannon(p_hist, q_hist) ** 2

output_dir = Path(".")
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(20, 15))
jsd_results = {}

morandi_pink = "#d8a5a3"  
morandi_silver = "#b0b0b0"  

for i, param in enumerate(parameters, 1):
    if param in data1.columns and param in data2.columns:
        p_samples = data1[param].dropna().values
        q_samples = data2[param].dropna().values
        jsd = calculate_jsd(p_samples, q_samples)
        jsd_results[param] = jsd
        plt.subplot(4, 2, i)
        sns.histplot(p_samples, bins=30, color=morandi_silver, kde=True, stat="density", label="Bilby", alpha=0.6)
        sns.histplot(q_samples, bins=30, color=morandi_pink, kde=True, stat="density", label="Amplfi", alpha=0.4)
        plt.title(f"({chr(96 + i)}) {param} Distribution - JSD: {jsd:.3f}")
        plt.xlabel(param)
        plt.ylabel("Density")
        plt.legend()

    else:
        print(f"Parameter {param} is missing in one of the files.")

save_path = output_dir / "parameter_comparison_JSD_5000.png"
plt.tight_layout()
plt.savefig(save_path)
plt.close()
print(f"All parameter comparison plots saved to {save_path}")

print("JSD Results for each parameter:")
for param, jsd_value in jsd_results.items():
    print(f"{param}: {jsd_value}")
