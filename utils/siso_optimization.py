import numpy as np
import matplotlib.pyplot as plt


# Hill Repression Function (IC50 model)
def inhibitory_hill(X, Bottom, Top, IC50, HillSlope):
    return Bottom + (Top - Bottom) / (1 + (IC50 / X) ** HillSlope)

# Hill Function Parameters for each TF
repressors = {
    "LacI_YQR": {
        "liganded": [-2214041.0, 2618.394333, 6.032e+17, -0.2572],
        "no_ligand": [-19.98, 2618.394333, 30.22, -0.9328],
    },
    "LacI_TAN": {
        "liganded": [723.8, 2627.360333, 1222, -1.238],
        "no_ligand": [26.9, 2627.360333, 33.63, -1.87],
    },
    "LacI_NAR": {
        "liganded": [196, 2215.056833, 396.2, -1.428],
        "no_ligand": [24.58, 2215.056833, 48.41, -1.922],
    },
    "LacI_HQN": {
        "liganded": [600.1, 7361.5495, 650.4, -1.214],
        "no_ligand": [29.69, 7361.5495, 27.47, -2.45],
    },
    "LacI_KSL": {
        "liganded": [184.8, 294.24755, 1459, -5.139],
        "no_ligand": [13.47, 294.24755, 54.06, -2.221],
    },
    "RbsR_YQR": {
        "liganded": [-1164597, 2618.394333, 26503170368, -0.4283],
        "no_ligand": [36.82, 2618.394333, 25.53, -2.839],
    },
    "RbsR_TAN": {
        "liganded": [552.3, 2627.360333, 316.2, -1.271],
        "no_ligand": [18.87, 2627.360333, 25.58, -2.265],
    },
    "RbsR_NAR": {
        "liganded": [98.2, 2215.056833, 117.9, -1.519],
        "no_ligand": [33.22, 2215.056833, 52.92, -1.838],
    },
    "RbsR_HQN": {
        "liganded": [-125.6, 7361.5495, 394, -1.17],
        "no_ligand": [35.3, 7361.5495, 24.1, -2.757],
    },
    "RbsR_KSL": {
        "liganded": [194, 294.24755, 734.1, -19.32],
        "no_ligand": [16.05, 294.24755, 39.61, -2.597],
    },
    "CelR_YQR": {
        "liganded": [2623.0, 2618.394333, 138.5, 211.3],
        "no_ligand": [0.007577, 2618.394333, 54.27, -0.981],
    },
    "CelR_TAN": {
        "liganded": [1014, 2627.360333, 705.1, -1.371],
        "no_ligand": [43.24, 2627.360333, 16.32, -4.634],
    },
    "CelR_NAR": {
        "liganded": [1437, 2215.056833, 645.9, -3.176],
        "no_ligand": [26.26, 2215.056833, 17.93, -2.308],
    },
    "CelR_HQN": {
        "liganded": [1047, 7361.5495, 460.7, -1.718],
        "no_ligand": [37.79, 7361.5495, 13.64, -4.329],
    },
    "CelR_KSL": {
        "liganded": [7.955, 294.24755, 1507, -2.182],
        "no_ligand": [21.23, 294.24755, 14.9, -3.258],
    }
}


anti_repressors = {
    "Anti-LacI_YQR": {
        "no_ligand": [-47306.0, 2618.394333, 7992.0, -2.638],
        "liganded": [-221.5, 2618.394333, 219.4, -0.9856],
    },
    "Anti-LacI_TAN": {
        "no_ligand": [1680.0, 2627.360333, 18498.0, -353.0],
        "liganded": [-59.39, 2627.360333, 234.5, -1.326],
    },
    "Anti-LacI_NAR": {
        "no_ligand": [-378.5, 2215.056833, 1818.0, -6.636],
        "liganded": [-4.179, 2215.056833, 318.0, -1.683],
    },
    "Anti-LacI_HQN": {
        "no_ligand": [-529.9, 7361.5495, 1384.0, -3.176],
        "liganded": [-171.6, 7361.5495, 85.58, -1.122],
    },
    "Anti-LacI_KSL": {
        "no_ligand": [-16757.0, 294.24755, 3263.0, -7.502],
        "liganded": [-35.46, 294.24755, 329.0, -1.085],
    },
    "Anti-RbsR_YQR": {
        "no_ligand": [17.31, 2618.394333, 1166.0, -1.604],
        "liganded": [52.75, 2618.394333, 53.21, -1.69],
    },
    "Anti-RbsR_TAN": {
        "no_ligand": [-54.74, 2627.360333, 413.2, -1.347],
        "liganded": [46.42, 2627.360333, 51.7, -2.22],
    },
    "Anti-RbsR_NAR": {
        "no_ligand": [82.21, 2215.056833, 417.9, -1.759],
        "liganded": [43.22, 2215.056833, 52.64, -2.653],
    },
    "Anti-RbsR_HQN": {
        "no_ligand": [-1004.0, 7361.5495, 561.9, -1.364],
        "liganded": [38.52, 7361.5495, 37.86, -2.222],
    },
    "Anti-RbsR_KSL": {
        "no_ligand": [-87.14, 294.24755, 1468.0, -2.223],
        "liganded": [24.7, 294.24755, 83.86, -3.163],
    },
    "Anti-CelR_YQR": {
        "no_ligand": [-14123293.0, 2618.394333, 3251870813, -0.6241],
        "liganded": [-15.47, 2618.394333, 95.77, -1.24],
    },
    "Anti-CelR_TAN": {
        "no_ligand": [79.16, 2627.360333, 102.1, -2.421],
        "liganded": [31.73, 2627.360333, 19.74, -98.7],
    },
    "Anti-CelR_NAR": {
        "no_ligand": [-7.528, 2215.056833, 161.1, -1.462],
        "liganded": [18.98, 2215.056833, 30.0, -1.946],
    },
    "Anti-CelR_HQN": {
        "no_ligand": [79.23, 7361.5495, 86.88, -2.401],
        "liganded": [38.41, 7361.5495, 19.65, -84.67],
    },
    "Anti-CelR_KSL": {
        "no_ligand": [22.45, 294.24755, 125.3, -1.855],
        "liganded": [12.15, 294.24755, 21.26, -3.563],
    },
}

# Combining into one dictionary
all_TFs = {**repressors, **anti_repressors}
TF_names = list(all_TFs.keys())

RBS_library = {
    'ATG_b': 1636.1414, 'ATG_c': 1279.3006, 'ATG_f': 730.46363, 'ATG_g': 752.41607,'GTG_a': 1119.6597, 'GTG_b': 748.72212,'GTG_d': 606.52384, 'GTG_e': 405.82922, 'GTG_g': 129.01393, 'GTG_h': 369.33754, 'GTG_i': 51.074853,
    'lacRBS': 325.23052
}
RBS_names = list(RBS_library.keys())

def is_repressor(tf_name):
    return tf_name in repressors



# Plotting function
def plot_tf(tf_name, rbs_name, UI_EU_IN):
    params = all_TFs[tf_name]
    X_vals_full = np.logspace(-2, 4, 2000)
    mask = X_vals_full <= 1700
    X_vals = X_vals_full[mask]

    Y_lig = np.clip(inhibitory_hill(X_vals_full, *params['liganded']), 1, None)[mask]
    Y_no = np.clip(inhibitory_hill(X_vals_full, *params['no_ligand']), 1, None)[mask]

    perf_X = X_vals
    perf_Y_lig = Y_lig
    perf_Y_no = Y_no

    if is_repressor(tf_name):
        performance = (perf_Y_lig - perf_Y_no) / perf_Y_no
        label_suffix = "ΔY / Y_no_ligand"
    else:
        performance = (perf_Y_no - perf_Y_lig) / perf_Y_lig
        label_suffix = "ΔY / Y_ligand"

    max_idx = np.argmax(performance)
    max_perf = performance[max_idx]
    optimal_X = perf_X[max_idx]
    y_lig_at_opt = perf_Y_lig[max_idx]
    y_no_at_opt = perf_Y_no[max_idx]

    # RBS EU_IN and output values
    rbs_val = RBS_library[rbs_name]
    y_lig_rbs = max(inhibitory_hill(rbs_val, *params['liganded']), 1)
    y_no_rbs = max(inhibitory_hill(rbs_val, *params['no_ligand']), 1)
    if is_repressor(tf_name):
        perf_rbs = (y_lig_rbs - y_no_rbs) / y_no_rbs
    else:
        perf_rbs = (y_no_rbs - y_lig_rbs) / y_lig_rbs

    # User input EU_IN and output values
    y_lig_ui = max(inhibitory_hill(UI_EU_IN, *params['liganded']), 1)
    y_no_ui = max(inhibitory_hill(UI_EU_IN, *params['no_ligand']), 1)
    if is_repressor(tf_name):
        perf_ui = (y_lig_ui - y_no_ui) / y_no_ui
    else:
        perf_ui = (y_no_ui - y_lig_ui) / y_lig_ui

    # Find closest RBS to optimal performance
    def get_perf(val):
        y_lig = max(inhibitory_hill(val, *params['liganded']), 1)
        y_no = max(inhibitory_hill(val, *params['no_ligand']), 1)
        if is_repressor(tf_name):
            return (y_lig - y_no) / y_no
        else:
            return (y_no - y_lig) / y_lig

    rbs_perf_diffs = {
        name: abs(get_perf(val) - max_perf)
        for name, val in RBS_library.items()
    }
    closest_rbs_name = min(rbs_perf_diffs, key=rbs_perf_diffs.get)
    closest_rbs_val = RBS_library[closest_rbs_name]

    # Plotting
    plt.figure(figsize=(12, 5), dpi=300)

    # Hill curves
    plt.subplot(1, 2, 1)
    plt.plot(X_vals, Y_no, label='No Ligand (OFF)', color='red')
    plt.plot(X_vals, Y_lig, label='With Ligand (ON)', color='green')
    plt.axvline(optimal_X, linestyle=':', color='gray', label=f"Max Perf @ X = {optimal_X:.2g}")
    plt.plot(optimal_X, y_lig_at_opt, 'ro')
    plt.plot(optimal_X, y_no_at_opt, 'ro')
    plt.axvline(rbs_val, color='purple', linestyle='--', label=f'RBS = {rbs_name}')
    plt.plot(rbs_val, y_lig_rbs, 'mo')
    plt.plot(rbs_val, y_no_rbs, 'mo')
    plt.axvline(UI_EU_IN, color='orange', linestyle='--', label=f'UI_EU_IN = {UI_EU_IN:.2f}')
    plt.plot(UI_EU_IN, y_lig_ui, 'o', color='orange')
    plt.plot(UI_EU_IN, y_no_ui, 'o', color='orange')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('EU_IN')
    plt.ylabel('EU_OUT')
    plt.title(f'Hill Curves: {tf_name}')
    plt.legend()
    plt.grid(True)

    # Performance plot
    plt.subplot(1, 2, 2)
    plt.plot(perf_X, performance, label=label_suffix, color='blue')
    plt.axvline(optimal_X, linestyle='--', color='gray')
    plt.plot(optimal_X, max_perf, 'ro', label=f"Max = {max_perf:.2f} at X = {optimal_X:.2g}")
    plt.axvline(rbs_val, color='purple', linestyle='--')
    plt.plot(rbs_val, perf_rbs, 'mo', label=f'Perf @ RBS = {perf_rbs:.2f}')
    plt.axvline(UI_EU_IN, color='orange', linestyle='--')
    plt.plot(UI_EU_IN, perf_ui, 'o', color='orange', label=f'Perf @ Input = {perf_ui:.2f}')
    plt.xscale('log')
    plt.xlabel('EU_IN')
    plt.ylabel('Performance')
    plt.title(f'Performance: {label_suffix}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Output text summary
    print(f"Closest RBS to optimal performance ({max_perf:.2f}): {closest_rbs_name} (EU_IN = {closest_rbs_val})")
    print(f"Selected RBS = {rbs_name} (EU_IN = {rbs_val})")
    print(f"→ ON = {y_lig_rbs:.2f}, OFF = {y_no_rbs:.2f}, Performance = {perf_rbs:.2f}")
    print(f"User Input EU_IN = {UI_EU_IN}")
    print(f"→ ON = {y_lig_ui:.2f}, OFF = {y_no_ui:.2f}, Performance = {perf_ui:.2f}")