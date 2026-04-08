# _erc_baseline_fao56_fully_automated_fixed_axes_

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap
import joblib

# ==========================================
# 0. CONFIGURATION & GLOBAL BOUNDARIES
# ==========================================
CONFIG = {
    "NUM_SAMPLES":   500_000,
    "BATCH_SIZE":    4096,
    "EPOCHS":        300,
    "PATIENCE":      30,
    "BASE_DIR":      r"C:\Users\FYEC_SKasa\Desktop\erc_ml_test", 
}

os.makedirs(CONFIG["BASE_DIR"], exist_ok=True)
DATA_PATH = os.path.join(CONFIG["BASE_DIR"], "erc_dataset.csv")

tf.config.set_visible_devices([], 'GPU')

# --- TEK MERKEZ: TÜM SINIRLAR BURADAN YÖNETİLİR ---

GLOBAL_TRAIN_BOUNDS = {
    "windspeed": (0.5,   10.0),
    "albedo":    (0.10,  0.40),
    "n":         (2.0,   12.0),
    "lat":       (-90.0, 90.0),
    "latmin":    (0.0,   59.0),
    "elevation": (0.0,   1000.0),
    "Tmax":      (15.0,  35.0),
    "Tmin":      (-5.0,  5.0),
    "rhum":      (0.25,  0.65),
    "J":         (1.0,   365.0),
}

GLOBAL_OOD_SCENARIOS = {
    "Extreme_Arid": {
        "windspeed": (5.0,   15.0),
        "albedo":    (0.25,  0.40),
        "n":         (11.0,  14.0),
        "lat":       (15.0,  35.0),
        "latmin":    (0.0,   59.0),
        "elevation": (0.0,   500.0),
        "Tmax":      (45.0,  55.0),
        "Tmin":      (25.0,  38.0),
        "rhum":      (0.1,  0.3),
        "J":         (150.0, 240.0),
    },
    "High_Altitude": {
        "windspeed": (8.0,   25.0),
        "albedo":    (0.60,  0.85),
        "n":         (2.0,   8.0),
        "lat":       (25.0,  50.0),
        "latmin":    (0.0,   59.0),
        "elevation": (3000.0,5500.0),
        "Tmax":      (-5.0,  5.0),
        "Tmin":      (-25.0, -8.0),
        "rhum":      (0.20,  0.40),
        "J":         (1.0,   365.0),
    },
    "Tropical_Monsoon": {
        "windspeed": (0.0,   2.0),
        "albedo":    (0.10,  0.16),
        "n":         (0.0,   3.0),
        "lat":       (-10.0, 10.0),
        "latmin":    (0.0,   59.0),
        "elevation": (0.0,   500.0),
        "Tmax":      (25.0,  35.0),
        "Tmin":      (24.0,  28.0),
        "rhum":      (0.80,  1.00),
        "J":         (1.0,   365.0),
    },
}

FEATURES_LIST = list(GLOBAL_TRAIN_BOUNDS.keys())

# ==========================================
# 1. PHYSICS — FAO-56 PENMAN-MONTEITH ERC
# ==========================================
def calculate_erc(windspeed, albedo, n, lat, latmin,
                  elevation, Tmax, Tmin, rhum, J):
    
    P          = 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.256
    Stefan     = 4.903e-9  
    Tmean      = (Tmax + Tmin) / 2.0
    
    lam        = 2.501 - 0.002361 * Tmean

    e_tmax     = 0.6108 * np.exp(17.27 * Tmax  / (237.3 + Tmax))
    e_tmin     = 0.6108 * np.exp(17.27 * Tmin  / (237.3 + Tmin))
    
    e_sat      = (e_tmax + e_tmin) / 2.0
    e_act      = e_sat * rhum

    gamma      = 0.0016286 * P / lam
    D          = e_sat - e_act 
    
    Delta      = 4098 * e_sat / (237.3 + Tmean) ** 2
    denom      = Delta + gamma * (1 + 0.34 * windspeed)

    delta_sun  = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    phi        = np.pi / 180 * (lat + latmin / 60.0)
    ws         = np.arccos(np.clip(-np.tan(phi) * np.tan(delta_sun), -1.0, 1.0))
    N          = 24 * ws / np.pi

    dr         = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    Isd        = 15.392 * dr * (ws * np.sin(phi) * np.sin(delta_sun)
                                + np.cos(phi) * np.cos(delta_sun) * np.sin(ws))

    Iscd       = np.where(N > 0, (0.25 + 0.5 * n / N) * Isd, 0.0)
    Sn         = Iscd * (1 - albedo)

    E_emis     = 0.34 - 0.14 * np.sqrt(np.maximum(e_act, 0))
    f_cloud    = np.where(Isd > 0, Iscd / Isd, 0.0)
    Ln         = -f_cloud * E_emis * Stefan * (Tmean + 273.15) ** 4 
    Rnet       = Sn + Ln

    Erc = ( (Delta / denom) * (Rnet / lam) 
          + (gamma / denom) * (900 / (Tmean + 273)) * windspeed * D )

    return np.round(Erc, 6)

# ==========================================
# 2. DATASET GENERATION
# ==========================================
def generate_dataset(num_samples):
    print(f"[DATA] Generating {num_samples:,} samples via Latin Hypercube...")
    t0 = time.time()

    bounds = GLOBAL_TRAIN_BOUNDS

    sampler = qmc.LatinHypercube(d=len(bounds), seed=42)
    raw     = sampler.random(n=num_samples)

    df = pd.DataFrame()
    for i, (col, (lo, hi)) in enumerate(bounds.items()):
        df[col] = qmc.scale(raw[:, i:i+1], [lo], [hi]).flatten()

    bad = df["Tmax"] <= df["Tmin"]
    df.loc[bad, "Tmax"] = df.loc[bad, "Tmin"] + np.random.uniform(2, 10, bad.sum())

    print("[DATA] Computing ERC targets...")
    df["Erc"] = calculate_erc(
        df.windspeed, df.albedo, df.n, df.lat, df.latmin,
        df.elevation, df.Tmax, df.Tmin, df.rhum, df.J
    )

    df = df.dropna().reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"[DATA] Done in {time.time()-t0:.1f}s  |  rows: {len(df):,}")
    return df

# ==========================================
# 3. MODEL
# ==========================================
def build_model(input_dim):
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(128, activation="swish")(inputs)
    x = tf.keras.layers.Dense(128, activation="swish")(x)
    x = tf.keras.layers.Dense(64,  activation="swish")(x)
    x = tf.keras.layers.Dense(32,  activation="swish")(x)
    out = tf.keras.layers.Dense(1, activation="linear")(x)

    model = tf.keras.Model(inputs, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model

# ==========================================
# 4. TRAINING FIGURE
# ==========================================
def plot_training(history, save_dir):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["loss"],     label="Train Loss", color="#2563EB", lw=2)
    ax.plot(epochs, history["val_loss"], label="Val Loss",   color="#DC2626", lw=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Huber Loss")
    ax.set_title("Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["mae"],     label="Train MAE", color="#2563EB", lw=2)
    ax.plot(epochs, history["val_mae"], label="Val MAE",   color="#DC2626", lw=2, linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MAE (mm/day)")
    ax.set_title("Mean Absolute Error")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "01_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

# ==========================================
# 5. IN-DISTRIBUTION TEST FIGURE
# ==========================================
def plot_indist_test(y_true, y_pred, save_dir):
    residuals = y_pred - y_true
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    bias = np.mean(residuals)

    fig = plt.figure(figsize=(16, 5))
    fig.suptitle(f"In-Distribution Test  |  MAE={mae:.4f}  R²={r2:.4f}  Bias={bias:.4f}",
                  fontsize=13, fontweight="bold")
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax = fig.add_subplot(gs[0])
    lim = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.scatter(y_true, y_pred, s=4, alpha=0.3, color="#2563EB", rasterized=True)
    ax.plot(lim, lim, "k--", lw=1.5, label="Perfect")
    ax.set_xlabel("True ERC (mm/day)"); ax.set_ylabel("Predicted ERC (mm/day)")
    ax.set_title("True vs Predicted"); ax.legend(); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1])
    ax.scatter(y_true, residuals, s=4, alpha=0.3, color="#059669", rasterized=True)
    ax.axhline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("True ERC (mm/day)"); ax.set_ylabel("Residual (mm/day)")
    ax.set_title("Residuals vs True"); ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2])
    ax.hist(residuals, bins=80, color="#7C3AED", edgecolor="none", alpha=0.8)
    ax.axvline(0, color="black", lw=1.5, linestyle="--")
    ax.set_xlabel("Residual (mm/day)"); ax.set_ylabel("Count")
    ax.set_title("Residual Distribution"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "02_indist_test.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

# ==========================================
# 6. OOD SCENARIOS GENERATION
# ==========================================
N_OOD = 5_000

def generate_ood():
    frames = {}
    for name, bounds in GLOBAL_OOD_SCENARIOS.items():
        sampler = qmc.LatinHypercube(d=len(bounds), seed=0)
        raw     = sampler.random(n=N_OOD)
        df = pd.DataFrame()
        for i, (col, (lo, hi)) in enumerate(bounds.items()):
            df[col] = qmc.scale(raw[:, i:i+1], [lo], [hi]).flatten()
        bad = df["Tmax"] <= df["Tmin"]
        df.loc[bad, "Tmax"] = df.loc[bad, "Tmin"] + 0.5
        df["Erc_true"] = calculate_erc(
            df.windspeed, df.albedo, df.n, df.lat, df.latmin,
            df.elevation, df.Tmax, df.Tmin, df.rhum, df.J
        )
        df = df.dropna()
        frames[name] = df
    return frames

# ==========================================
# 7. OOD FIGURES
# ==========================================
COLORS = {
    "Extreme_Arid":     "#e11d48", # Rose-600
    "High_Altitude":    "#2563eb", # Blue-600
    "Tropical_Monsoon": "#059669", # Emerald-600
}
TRAIN_COLOR = "#4b5563" # Gray-600

def plot_ood_overview(ood_frames, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("OOD Test — True vs Predicted ERC per Scenario",
                  fontsize=14, fontweight="bold")

    for ax, (name, df) in zip(axes, ood_frames.items()):
        mae = mean_absolute_error(df["Erc_true"], df["Erc_pred"])
        r2  = r2_score(df["Erc_true"], df["Erc_pred"])
        col = COLORS[name]

        lim = [min(df["Erc_true"].min(), df["Erc_pred"].min()),
               max(df["Erc_true"].max(), df["Erc_pred"].max())]
        ax.scatter(df["Erc_true"], df["Erc_pred"],
                    s=8, alpha=0.6, color=col, rasterized=True)
        ax.plot(lim, lim, "k--", lw=1.5)
        ax.set_title(f"{name.replace('_',' ')}\nMAE={mae:.3f}  R²={r2:.3f}", fontsize=11)
        ax.set_xlabel("True ERC (mm/day)")
        ax.set_ylabel("Predicted ERC (mm/day)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "03_ood_true_vs_pred.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

def plot_ood_residuals(ood_frames, save_dir):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("OOD Residual Distributions", fontsize=13, fontweight="bold")

    for name, df in ood_frames.items():
        residuals = df["Erc_pred"] - df["Erc_true"]
        ax.hist(residuals, bins=80, alpha=0.65, color=COLORS[name],
                label=f"{name.replace('_',' ')}  (bias={residuals.mean():.2f})",
                edgecolor="none")

    ax.axvline(0, color="black", lw=2, linestyle="--")
    ax.set_xlabel("Residual (mm/day)"); ax.set_ylabel("Count")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "04_ood_residuals.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

def plot_ood_mae_bar(ood_frames, indist_mae, save_dir):
    labels = ["In-Distribution"] + [n.replace("_", "\n") for n in ood_frames]
    maes   = [indist_mae] + [
        mean_absolute_error(df["Erc_true"], df["Erc_pred"])
        for df in ood_frames.values()
    ]
    bar_colors = [TRAIN_COLOR] + list(COLORS.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("MAE Comparison: In-Distribution vs OOD Scenarios",
                  fontsize=13, fontweight="bold")

    bars = ax.bar(labels, maes, color=bar_colors, edgecolor="white", width=0.55)
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("MAE (mm/day)", fontsize=12)
    ax.set_ylim(0, max(maes) * 1.25)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "05_mae_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

def plot_ood_error_vs_feature(ood_frames, save_dir):
    feature_map = {
        "Extreme_Arid":    ("Tmax",      "Tmax (°C)"),
        "High_Altitude":   ("elevation", "Elevation (m)"),
        "Tropical_Monsoon":("rhum",      "Relative Humidity"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("OOD — Absolute Error vs Key OOD Feature",
                  fontsize=14, fontweight="bold")

    for ax, (name, df) in zip(axes, ood_frames.items()):
        feat_col, feat_label = feature_map[name]
        abs_err = np.abs(df["Erc_pred"] - df["Erc_true"])
        ax.scatter(df[feat_col], abs_err, s=6, alpha=0.6,
                   color=COLORS[name], rasterized=True)
        ax.set_xlabel(feat_label)
        ax.set_ylabel("Absolute Error (mm/day)")
        ax.set_title(name.replace("_", " "))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "06_ood_error_vs_feature.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

# ==========================================
# 8. TRAIN vs OOD BOUNDARY FIGURE
# ==========================================
def plot_boundary_figure(df_train, ood_frames, save_dir):
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.suptitle("The Neural Network's Domain Limitation\nExtrapolation Beyond Training Data → Extreme Errors",
                  fontsize=15, fontweight="bold")

    cmap_density = LinearSegmentedColormap.from_list(
    "soft_gray_to_blue",
    ["#e5e7eb", "#60a5fa"] 
    )
    
    cmap_error = LinearSegmentedColormap.from_list(
    "dark_yellow_to_red",
    ["#b59f00", "#ef4444"] 
    )

    train_density = ax.hexbin(
        df_train["Tmax"], df_train["rhum"], 
        gridsize=50, cmap=cmap_density, norm=LogNorm(), 
        alpha=0.6, zorder=1, rasterized=True
    )
    cbar_density = plt.colorbar(train_density, ax=ax, pad=0.01)
    cbar_density.set_label("Training Data Density (Log Count)", fontsize=10)

    tmax_lo, tmax_hi = GLOBAL_TRAIN_BOUNDS["Tmax"]
    rhum_lo, rhum_hi = GLOBAL_TRAIN_BOUNDS["rhum"]
    
    rect = plt.Rectangle(
        (tmax_lo, rhum_lo),
        tmax_hi - tmax_lo,
        rhum_hi - rhum_lo,
        linewidth=3.0, edgecolor="black", facecolor="none",
        linestyle="-", label="Training Data Boundary", zorder=3
    )
    ax.add_patch(rect)

    markers = {
        "Extreme_Arid":     "^",
        "High_Altitude":    "s",
        "Tropical_Monsoon": "o",
    }
    
    error_scatter = None
    for name, df_ood in ood_frames.items():
        abs_err = np.abs(df_ood["Erc_pred"] - df_ood["Erc_true"])
        error_scatter = ax.scatter(
            df_ood["Tmax"], df_ood["rhum"],
            c=abs_err,
            cmap=cmap_error,  
            s=60, alpha=0.9, 
            marker=markers[name],
            label=f"{name.replace('_', ' ')}  (Scenario)",
            vmin=0, vmax=6,
            zorder=4, edgecolor="white", linewidth=0.3, rasterized=True
        )

    cbar_error = plt.colorbar(error_scatter, ax=ax, pad=0.03)
    cbar_error.set_label("Out-of-Distribution Error (MAE mm/day)", fontsize=12, fontweight="bold")
    
    boundary_legend = plt.Line2D([0], [0], color='black', linewidth=3.0, linestyle='-', label="Training Data Boundary")
    
    current_handles, current_labels = ax.get_legend_handles_labels()
    ax.legend(handles=current_handles + [boundary_legend], fontsize=10, loc="upper right", facecolor='white', framealpha=0.95)
    
    ax.set_xlabel("Tmax (°C) [Primary Temperature Feature]", fontsize=12)
    ax.set_ylabel("Relative Humidity [Primary Dryness Feature]", fontsize=12)

    # ---------------------------------------------------------
    # GÜNCELLEME: EKSEN SINIRLARI TÜM VERİYİ KAPSAYACAK ŞEKİLDE SABİTLENDİ
    # High Altitude -5'e kadar iniyor, Extreme Arid 52'ye kadar çıkıyor.
    # Tropical Monsoon nem 1.00'a çıkıyor, Extreme Arid 0.05'e iniyor.
    # Çerçevenin dışında hiçbir veri kalmaması için eksenler genişletildi:
    ax.set_xlim(-10, 60) 
    ax.set_ylim(0.0, 1.05)
    # ---------------------------------------------------------

    ax.grid(True, alpha=0.3, zorder=0)

    plt.tight_layout()
    path = os.path.join(save_dir, "07_boundary_overview.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

# ==========================================
# 9. INPUT FEATURE RANGES SUMMARY FIGURE
# ==========================================
def plot_input_ranges(save_dir):
    FEATURES_META = {
        "windspeed":  ("Wind Speed",       "m/s"),
        "albedo":     ("Albedo",           "-"),
        "n":          ("Sunshine Hours",    "hr"),
        "lat":        ("Latitude",          "°"),
        "latmin":     ("Lat. Minutes",      "'"),
        "elevation":  ("Elevation",         "m"),
        "Tmax":       ("Tmax",              "°C"),
        "Tmin":       ("Tmin",              "°C"),
        "rhum":       ("Rel. Humidity",     "-"),
        "J":          ("Day of Year",       "day"),
    }

    features   = FEATURES_LIST 
    n_features = len(features)

    fig, axes = plt.subplots(n_features, 1, figsize=(14, n_features * 1.05))
    fig.suptitle("Input Feature Ranges: Training Domain vs OOD Scenarios\nExtrapolation Occurs Outside Training Boundary Vertical Lines",
                  fontsize=14, fontweight="bold", y=1.01)

    bar_height  = 0.22
    y_train     = 0.0
    y_positions = {"Extreme_Arid": -0.25, "High_Altitude": -0.50, "Tropical_Monsoon": -0.75}

    for ax, feat in zip(axes, features):
        label, unit  = FEATURES_META[feat]
        
        tr_lo, tr_hi = GLOBAL_TRAIN_BOUNDS[feat]

        ax.barh(y_train, tr_hi - tr_lo, left=tr_lo,
                height=bar_height, color=TRAIN_COLOR, alpha=0.9,
                label="Training Domain" if feat == features[0] else "", zorder=2)

        for scenario, y_pos in y_positions.items():
            od_lo, od_hi = GLOBAL_OOD_SCENARIOS[scenario][feat]
            color        = COLORS[scenario]

            ax.barh(y_pos, od_hi - od_lo, left=od_lo,
                    height=bar_height, color=color, alpha=0.65,
                    label=scenario.replace("_", " ") if feat == features[0] else "",
                    zorder=2)

            if od_lo < tr_lo:
                extrapol_width = min(tr_lo, od_hi) - od_lo
                ax.barh(y_pos, extrapol_width, left=od_lo,
                        height=bar_height, color=color, alpha=1.0,
                        hatch="////", edgecolor="#1e293b", linewidth=1.5, zorder=3)
            
            if od_hi > tr_hi:
                extrapol_width = od_hi - max(od_lo, tr_hi)
                ax.barh(y_pos, extrapol_width, left=max(od_lo, tr_hi),
                        height=bar_height, color=color, alpha=1.0,
                        hatch="////", edgecolor="#1e293b", linewidth=1.5, zorder=3)

        ax.axvline(tr_lo, color="black", lw=1.2, linestyle="--", alpha=0.8, zorder=4)
        ax.axvline(tr_hi, color="black", lw=1.2, linestyle="--", alpha=0.8, zorder=4)

        if feat == "Tmax":
            ax.set_xlim(0.0, 60.0) 
        elif feat == "rhum":
            ax.set_xlim(0.0, 1.0) 

        ax.set_ylabel(f"{label}\n({unit})", fontsize=8.5, rotation=0,
                      ha="right", va="center", labelpad=60)
        ax.set_yticks([]) 
        ax.grid(True, axis="x", alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)

    handles = [
        plt.Rectangle((0,0),1,1, color=TRAIN_COLOR, alpha=0.9, label="Training Domain"),
        plt.Rectangle((0,0),1,1, color=COLORS["Extreme_Arid"], alpha=0.8, label="Extreme Arid (OOD)"),
        plt.Rectangle((0,0),1,1, color=COLORS["High_Altitude"], alpha=0.8, label="High Altitude (OOD)"),
        plt.Rectangle((0,0),1,1, color=COLORS["Tropical_Monsoon"], alpha=0.8, label="Tropical Monsoon (OOD)"),
        plt.Rectangle((0,0),1,1, facecolor="none", hatch="////", edgecolor="#1e293b", linewidth=1.5, label="OOD Extrapolation (Bozunma Bölgesi)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=9, bbox_to_anchor=(0.5, -0.02),
               frameon=True, edgecolor="gray")

    plt.tight_layout()
    path = os.path.join(save_dir, "08_input_ranges_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[FIG] Saved: {path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    df = generate_dataset(CONFIG["NUM_SAMPLES"])

    X = df[FEATURES_LIST].values
    y = df["Erc"].values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.10, random_state=42
    )

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_train).astype(np.float32)
    X_te_sc  = scaler.transform(X_test).astype(np.float32)
    y_tr     = y_train.astype(np.float32)
    y_te     = y_test.astype(np.float32)

    model = build_model(len(FEATURES_LIST))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=CONFIG["PATIENCE"],
            restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=10,
            min_lr=1e-6, verbose=1),
    ]

    print("\n[TRAIN] Starting training...")
    t0 = time.time()
    hist = model.fit(
        X_tr_sc, y_tr,
        validation_split=0.10,
        epochs=CONFIG["EPOCHS"],
        batch_size=CONFIG["BATCH_SIZE"],
        callbacks=callbacks,
        verbose=2,
    )
    print(f"[TRAIN] Finished in {time.time()-t0:.1f}s")
    
    model_path = os.path.join(CONFIG["BASE_DIR"], "baseline_nn_erc.keras")
    model.save(model_path)                             
    print(f"[MODEL] Kaydedildi: {model_path}")
    
    scaler_path = os.path.join(CONFIG["BASE_DIR"], "scaler.pkl")
    joblib.dump(scaler, scaler_path)                
    print(f"[SCALER] Kaydedildi: {scaler_path}")
    
    plot_training(hist.history, CONFIG["BASE_DIR"])

    print("\n[TEST] Predicting on In-Distribution test set...")
    y_pred_test = model.predict(X_te_sc, batch_size=CONFIG["BATCH_SIZE"]).flatten()
    y_true_test = y_te.flatten()

    indist_mae = mean_absolute_error(y_true_test, y_pred_test)
    indist_r2  = r2_score(y_true_test, y_pred_test)
    print(f"[TEST] In-Distribution Results: MAE={indist_mae:.4f}  R²={indist_r2:.4f}")

    plot_indist_test(y_true_test, y_pred_test, CONFIG["BASE_DIR"])

    print("\n[OOD] Generating OOD scenarios (Extreme Arid, Altitude, Monsoon)...")
    ood_raw = generate_ood()

    ood_frames = {}
    print("\n[OOD] Running Out-of-Distribution Predictions:")
    for name, df_ood in ood_raw.items():
        X_ood_sc = scaler.transform(df_ood[FEATURES_LIST].values).astype(np.float32)
        
        df_ood["Erc_pred"] = model.predict(
            X_ood_sc, batch_size=CONFIG["BATCH_SIZE"]).flatten()
        
        mae = mean_absolute_error(df_ood["Erc_true"], df_ood["Erc_pred"])
        r2  = r2_score(df_ood["Erc_true"], df_ood["Erc_pred"])
        print(f"  {name:<22} Results: MAE={mae:.4f}  R²={r2:.4f}")
        ood_frames[name] = df_ood

    plot_ood_overview(ood_frames, CONFIG["BASE_DIR"])
    plot_ood_residuals(ood_frames, CONFIG["BASE_DIR"])
    plot_ood_mae_bar(ood_frames, indist_mae, CONFIG["BASE_DIR"])
    plot_ood_error_vs_feature(ood_frames, CONFIG["BASE_DIR"])
    plot_boundary_figure(df, ood_frames, CONFIG["BASE_DIR"])  
    plot_input_ranges(CONFIG["BASE_DIR"])   

    print(f"\n[DONE] All figures successfully saved to: {CONFIG['BASE_DIR']}")
    
    
def create_max_error_journey_video(model, scaler, save_dir):
    from matplotlib.animation import FFMpegWriter, FuncAnimation
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    from scipy.stats import qmc
    import pandas as pd
    import numpy as np

    print("[VIDEO] 10 Boyutlu Maksimum Hata odaklı video hazırlanıyor...")
    save_path = os.path.join(save_dir, "erc_10D_max_error_journey.mp4")
    
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_xlim(-15, 65)
    ax.set_ylim(-0.05, 1.1)
    ax.set_title("Model Stress Test, Point Visiting for Global Maximum Errors", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Tmax (°C)", fontsize=12)
    ax.set_ylabel("Relative Humidity", fontsize=12)

    # --- 1. 2D GRAFİK İÇİN KUTU ÇİZİMLERİ ---
    plot_zones = {
        "TRAIN ZONE":       {"range": (GLOBAL_TRAIN_BOUNDS["Tmax"], GLOBAL_TRAIN_BOUNDS["rhum"]), "color": "green", "ls": "--"},
        "EXTREME ARID":     {"range": (GLOBAL_OOD_SCENARIOS["Extreme_Arid"]["Tmax"], GLOBAL_OOD_SCENARIOS["Extreme_Arid"]["rhum"]),  "color": "#e11d48", "ls": "-"},
        "TROPICAL MONSOON": {"range": (GLOBAL_OOD_SCENARIOS["Tropical_Monsoon"]["Tmax"], GLOBAL_OOD_SCENARIOS["Tropical_Monsoon"]["rhum"]),  "color": "#059669", "ls": "-"},
        "HIGH ALTITUDE":    {"range": (GLOBAL_OOD_SCENARIOS["High_Altitude"]["Tmax"], GLOBAL_OOD_SCENARIOS["High_Altitude"]["rhum"]), "color": "#2563eb", "ls": "-"}
    }

    for name, info in plot_zones.items():
        (t_min, t_max), (r_min, r_max) = info["range"]
        ax.add_patch(patches.Rectangle((t_min, r_min), t_max-t_min, r_max-r_min, 
                                       lw=3, edgecolor=info["color"], facecolor=info["color"], alpha=0.1, ls=info["ls"]))
        ax.text(t_min, r_max + 0.01, name, color=info["color"], fontweight='bold', fontsize=9)

    # --- 2. 10 BOYUTLU EN KÖTÜ NOKTAYI BULAN FONKSİYON ---
    def get_worst_10d_combination(bounds_dict, samples=5000):
        sampler = qmc.LatinHypercube(d=len(FEATURES_LIST), seed=42)
        raw = sampler.random(n=samples)
        
        df_temp = pd.DataFrame()
        for i, feat in enumerate(FEATURES_LIST):
            lo, hi = bounds_dict[feat]
            df_temp[feat] = qmc.scale(raw[:, i:i+1], [lo], [hi]).flatten()
            
        # Fiziksel kural kontrolü
        bad = df_temp["Tmax"] <= df_temp["Tmin"]
        df_temp.loc[bad, "Tmax"] = df_temp.loc[bad, "Tmin"] + 0.5
        
        # Gerçek ERC hesabı
        true_vals = calculate_erc(df_temp.windspeed, df_temp.albedo, df_temp.n, df_temp.lat, df_temp.latmin,
                                  df_temp.elevation, df_temp.Tmax, df_temp.Tmin, df_temp.rhum, df_temp.J)
        
        # Model tahmini
        X_sc = scaler.transform(df_temp[FEATURES_LIST].values).astype(np.float32)
        pred_vals = model.predict(X_sc, batch_size=2048, verbose=0).flatten()
        
        # Maksimum hatayı bul ve o satırı (10 parametreyi) döndür
        errors = np.abs(pred_vals - true_vals)
        max_idx = np.argmax(errors)
        return df_temp[FEATURES_LIST].values[max_idx]

    # Başlangıç noktası olarak Train setinin tam merkezini (ortalama koşullar) alalım
    center_train_10d = np.array([(GLOBAL_TRAIN_BOUNDS[f][0] + GLOBAL_TRAIN_BOUNDS[f][1])/2.0 for f in FEATURES_LIST])

    print("        [INFO] Her bölgedeki en kötü 10 parametreli kombinasyonlar aranıyor...")
    wp_arid    = get_worst_10d_combination(GLOBAL_OOD_SCENARIOS["Extreme_Arid"])
    wp_monsoon = get_worst_10d_combination(GLOBAL_OOD_SCENARIOS["Tropical_Monsoon"])
    wp_alt     = get_worst_10d_combination(GLOBAL_OOD_SCENARIOS["High_Altitude"])

    # --- 3. 10 BOYUTLU YOLCULUK ROTASI ---
    waypoints_10d = [center_train_10d, wp_arid, wp_monsoon, wp_alt, center_train_10d]
    
    total_frames = 600
    steps_per_leg = total_frames // (len(waypoints_10d) - 1)
    
    # Tüm kareler (frame) için 10 boyutlu matrisi doldur
    all_frames_10d = []
    for i in range(len(waypoints_10d) - 1):
        leg_frames = np.linspace(waypoints_10d[i], waypoints_10d[i+1], steps_per_leg)
        all_frames_10d.extend(leg_frames)
        
    all_frames_10d = np.array(all_frames_10d) # Boyut: (600, 10)

    # --- 4. HIZLI İŞLEM: TÜM TAHMİNLERİ TEK SEFERDE YAP ---
    # Videonun takılmaması için for döngüsünün dışında tüm hesaplamaları peşin yapıyoruz.
    df_anim = pd.DataFrame(all_frames_10d, columns=FEATURES_LIST)
    
    true_anim = calculate_erc(df_anim.windspeed, df_anim.albedo, df_anim.n, df_anim.lat, df_anim.latmin,
                              df_anim.elevation, df_anim.Tmax, df_anim.Tmin, df_anim.rhum, df_anim.J)
                              
    X_anim_sc = scaler.transform(all_frames_10d).astype(np.float32)
    pred_anim = model.predict(X_anim_sc, batch_size=2048, verbose=0).flatten()
    
    errors_anim = np.abs(pred_anim - true_anim)

    # Görselleştirme indeksleri (Topun konumunu bulmak için)
    tmax_idx = FEATURES_LIST.index("Tmax")
    rhum_idx = FEATURES_LIST.index("rhum")

    # --- 5. ANİMASYON OBJELERİ ---
    dot, = ax.plot([], [], 'o', markersize=18, markeredgecolor='white', markeredgewidth=2, zorder=30)
    err_text = ax.text(0, 0, '', fontsize=13, fontweight='bold', ha='center', va='bottom', zorder=31)
    
    norm = mcolors.Normalize(vmin=0, vmax=2)
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["#22c55e", "#fbbf24", "#ef4444"])

    def update(frame):
        # 10 boyutlu matristen sadece 2 ekseni (Tmax ve Rhum) grafiğe basmak için çekiyoruz
        tmax = all_frames_10d[frame, tmax_idx]
        rhum = all_frames_10d[frame, rhum_idx]
        
        # Önceden hesaplanmış hatayı çek
        abs_err = errors_anim[frame]
        
        color = cmap(norm(min(abs_err, 2)))
        
        dot.set_data([tmax], [rhum])
        dot.set_color(color)
        
        err_text.set_position((tmax, rhum + 0.04))
        err_text.set_text(f"MA_Error: {abs_err:.2f}")
        err_text.set_color(color)

        return dot, err_text

    ani = FuncAnimation(fig, update, frames=len(all_frames_10d), interval=60, blit=True)
    writer = FFMpegWriter(fps=23, bitrate=7000)
    ani.save(save_path, writer=writer)
    plt.close()
    print(f"[DONE] 10 Boyutlu Video hazır! Konum: {save_path}")

# Not: Bu fonksiyonu ana kodunun en altına, model ve scaler eğitildikten sonra çağırabilirsin.
create_max_error_journey_video(model, scaler, CONFIG["BASE_DIR"])

print(f"\n[DONE] Tüm işlemler tamamlandı.")