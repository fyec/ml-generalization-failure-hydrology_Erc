# _pinn_combination_tester_v2.py

import os
import time
import itertools
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import qmc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import joblib

# ==========================================
# CONFIGURATION SECTION
# ==========================================
CONFIG = {
    "NUM_SAMPLES": 2000000,           
    "BATCH_SIZE": 8192,              
    "EPOCHS": 500,                    
    "PATIENCE": 50,                  
    "BASE_DIR": r"C:\Users\fyece\Desktop\nn_test_w_erc", 
    "OOD_SAMPLES_PER_SCENARIO": 5000,
    "LAMBDA_PHYSICS": 0.5             
}

os.makedirs(CONFIG["BASE_DIR"], exist_ok=True)
DATA_PATH = os.path.join(CONFIG["BASE_DIR"], "erc_training_dataset_pinn.csv")

# ==========================================
# 1. MATHEMATICAL FUNCTIONS
# ==========================================
def calculate_erc_vectorized(windspeed, albedo, n, lat, latmin, elevation, Tmax, Tmin, rhum, J):
    P = 101.3 * ((293 - 0.0065 * elevation) / 293)**5.256
    Stefboltzcons = 4.903 * 10**-9
    Tmean = Tmax / 2 + Tmin / 2
    λ = 2.501 - 0.0002361 * Tmean
    
    e_tmax = 0.6108 * np.exp((17.27 * Tmax) / (237.3 + Tmax))
    e_tmin = 0.6108 * np.exp((17.27 * Tmin) / (237.3 + Tmin))
    e_sat = (e_tmax / 2 + e_tmin / 2) * rhum
    
    γ = 0.0016286 * P / λ
    D = (e_tmax / 2 + e_tmin / 2) * (100 - rhum * 100) / 100
    Δ = 4098 * e_sat / ((237.3 + Tmean)**2)
    
    γmod = γ * (1 + 0.33 * windspeed)
    δ = 0.4093 * np.sin(2 * np.pi * J / 365 - 1.405)
    φ = np.pi / 180 * (lat + latmin / 60)
    
    ws = np.arccos(np.clip(-np.tan(φ) * np.tan(δ), -1.0, 1.0))
    N = 24 * ws / np.pi
    
    dr = 1 + 0.033 * np.cos(2 * np.pi * J / 365)
    Isd = 15.392 * dr * (ws * np.sin(φ) * np.sin(δ) + np.cos(φ) * np.cos(δ) * np.sin(ws))
    
    Iscd = np.where(N > 0, (0.25 + 0.5 * n / N) * Isd, 0)
    Sn = Iscd * (1 - albedo)
    
    E = 0.34 - 0.14 * (e_sat**0.5)
    f = np.where(Isd > 0, Iscd / Isd, 0)
    Ln = -f * E * Stefboltzcons * ((Tmean + 273.15)**4) / λ
    
    Rnet = Sn + Ln
    G = 0 
    
    Erc = (Δ / (Δ + γmod)) * (Rnet - G) + ((γ / (Δ + γmod)) * (900 / (Tmean + 275))) * windspeed * D
    
    return np.round(Erc, 2), Rnet, Tmean

# ==========================================
# 2. DATA GENERATION
# ==========================================
def generate_standard_training_set(num_samples):
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH}...")
        return pd.read_csv(DATA_PATH)

    print(f"\n--- STEP 1: Generating {num_samples:,} Training Samples ---")
    start_time = time.time()
    
    bounds = {
        "windspeed": (0.5, 8.0), "albedo": (0.15, 0.30), "n": (2.0, 12.0), 
        "lat": (-50.0, 50.0), "latmin": (0.0, 59.0), "elevation": (0.0, 2500.0), 
        "Tmax": (10.0, 38.0), "Tmin": (-5.0, 22.0), "rhum": (0.30, 0.90), "J": (1.0, 365.0)
    }
    
    sampler = qmc.LatinHypercube(d=len(bounds))
    sample = sampler.random(n=num_samples)
    
    df = pd.DataFrame()
    for i, (var, (lower, upper)) in enumerate(bounds.items()):
        df[var] = qmc.scale(sample[:, i].reshape(-1, 1), [lower], [upper]).flatten()
        
    mask_invalid_temp = df['Tmax'] <= df['Tmin']
    df.loc[mask_invalid_temp, 'Tmax'] = df.loc[mask_invalid_temp, 'Tmin'] + np.random.uniform(2, 10, size=mask_invalid_temp.sum())
    df['n'] = np.where(df['n'] > 14, 14, df['n']) 

    print("Calculating targets (Erc, Rnet, Tmean)...")
    df['Erc_target'], df['Rnet_true'], df['Tmean_true'] = calculate_erc_vectorized(
        df['windspeed'], df['albedo'], df['n'], df['lat'], df['latmin'], 
        df['elevation'], df['Tmax'], df['Tmin'], df['rhum'], df['J']
    )
    
    df = df.dropna()
    df.to_csv(DATA_PATH, index=False)
    print(f"Data generation complete in {time.time() - start_time:.2f} seconds.")
    return df

# ==========================================
# 3. PINN ARCHITECTURE
# ==========================================
class PINNModel(tf.keras.Model):
    def __init__(self, inputs, outputs, flags, **kwargs):
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.use_boundary = flags['boundary']
        self.use_conservation = flags['conservation']
        self.use_monotonicity = flags['monotonicity']
        self.lambda_phys = CONFIG["LAMBDA_PHYSICS"]

    def train_step(self, data):
        x, y_true_multi = data
        Erc_true = y_true_multi[:, 0:1]
        Rnet_true = y_true_multi[:, 1:2]
        Tmean_true = y_true_multi[:, 2:3]

        with tf.GradientTape() as tape:
            with tf.GradientTape() as physics_tape:
                physics_tape.watch(x)
                preds = self(x, training=True)
                Erc_pred = preds[:, 0:1]
                H_pred = preds[:, 1:2] 
            
            loss_data = tf.reduce_mean(tf.keras.losses.huber(Erc_true, Erc_pred))
            
            loss_boundary = tf.constant(0.0)
            loss_conservation = tf.constant(0.0)
            loss_monotonicity = tf.constant(0.0)
            
            if self.use_boundary:
                loss_boundary = tf.reduce_mean(tf.square(tf.maximum(0.0, -Erc_pred)))
                
            if self.use_conservation:
                lam = 2.501 - 0.0002361 * Tmean_true
                energy_balance_error = Rnet_true - (lam * Erc_pred) - H_pred
                loss_conservation = tf.reduce_mean(tf.square(energy_balance_error))
                
            if self.use_monotonicity:
                grads = physics_tape.gradient(Erc_pred, x)
                dE_dTmax = grads[:, 6:7]
                loss_monotonicity = tf.reduce_mean(tf.square(tf.maximum(0.0, -dE_dTmax)))
                
            loss_physics = loss_boundary + loss_conservation + loss_monotonicity
            total_loss = loss_data + (self.lambda_phys * loss_physics)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        self.compiled_metrics.update_state(Erc_true, Erc_pred)
        return {"loss": total_loss, "mae": self.compiled_metrics.metrics[0].result()}

    def test_step(self, data):
        x, y_true_multi = data
        Erc_true = y_true_multi[:, 0:1]
        preds = self(x, training=False)
        Erc_pred = preds[:, 0:1]
        loss_data = tf.reduce_mean(tf.keras.losses.huber(Erc_true, Erc_pred))
        self.compiled_metrics.update_state(Erc_true, Erc_pred)
        return {"loss": loss_data, "mae": self.compiled_metrics.metrics[0].result()}

def build_pinn_model(input_dim, flags):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(256, activation='relu')(inputs)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    outputs = tf.keras.layers.Dense(2, activation='linear')(x) # Erc_pred, H_pred
    
    model = PINNModel(inputs=inputs, outputs=outputs, flags=flags)
    # clipnorm eklendi: Fiziksel gradyanlar patlarsa diye sınırı korur.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0), metrics=['mae'])
    return model

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================
def generate_ood_data():
    scenarios = {
        "Extreme_Arid": {"windspeed": (5, 15), "albedo": (0.25, 0.35), "n": (11, 14), "lat": (15, 35), "latmin": (0, 60), "elevation": (-100, 200), "Tmax": (40, 52), "Tmin": (25, 35), "rhum": (0.05, 0.15), "J": (150, 240)},
        "High_Altitude": {"windspeed": (10, 25), "albedo": (0.60, 0.85), "n": (2, 8), "lat": (25, 50), "latmin": (0, 60), "elevation": (3000, 5000), "Tmax": (-5, 10), "Tmin": (-25, -10), "rhum": (0.20, 0.50), "J": (1, 365)},
        "Tropical_Monsoon": {"windspeed": (0, 2), "albedo": (0.10, 0.15), "n": (0, 3), "lat": (0, 10), "latmin": (0, 60), "elevation": (0, 500), "Tmax": (28, 33), "Tmin": (24, 27), "rhum": (0.90, 1.00), "J": (1, 365)}
    }
    dfs = []
    for name, bounds in scenarios.items():
        sampler = qmc.LatinHypercube(d=len(bounds))
        sample = sampler.random(n=CONFIG["OOD_SAMPLES_PER_SCENARIO"])
        scaled_data = {var: qmc.scale(sample[:, i].reshape(-1, 1), [lower], [upper]).flatten() for i, (var, (lower, upper)) in enumerate(bounds.items())}
        df_scenario = pd.DataFrame(scaled_data)
        df_scenario['Tmax'] = np.where(df_scenario['Tmax'] <= df_scenario['Tmin'], df_scenario['Tmin'] + 0.1, df_scenario['Tmax'])
        df_scenario['Scenario'] = name
        df_scenario['True_Erc'], _, _ = calculate_erc_vectorized(
            df_scenario['windspeed'], df_scenario['albedo'], df_scenario['n'], 
            df_scenario['lat'], df_scenario['latmin'], df_scenario['elevation'], 
            df_scenario['Tmax'], df_scenario['Tmin'], df_scenario['rhum'], df_scenario['J']
        )
        dfs.append(df_scenario.dropna())
    return pd.concat(dfs, ignore_index=True), scenarios

# ==========================================
# MAIN EXECUTION & ITERATOR
# ==========================================
if __name__ == "__main__":
    df_train = generate_standard_training_set(CONFIG["NUM_SAMPLES"])
    ood_df, scenarios = generate_ood_data()
    
    X = df_train[['windspeed', 'albedo', 'n', 'lat', 'latmin', 'elevation', 'Tmax', 'Tmin', 'rhum', 'J']]
    y = df_train[['Erc_target', 'Rnet_true', 'Tmean_true']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    scaler = StandardScaler()
    X_train_tsr = tf.convert_to_tensor(scaler.fit_transform(X_train), dtype=tf.float32)
    y_train_tsr = tf.convert_to_tensor(y_train.values, dtype=tf.float32)
    
    features = ['windspeed', 'albedo', 'n', 'lat', 'latmin', 'elevation', 'Tmax', 'Tmin', 'rhum', 'J']
    X_ood_tsr = tf.convert_to_tensor(scaler.transform(ood_df[features]), dtype=tf.float32)

    results_table = []

    combinations = list(itertools.product([True, False], repeat=3))
    
    for b, c, m in combinations:
        flags = {'boundary': b, 'conservation': c, 'monotonicity': m}
        config_name = f"B-{str(b)[0]}_C-{str(c)[0]}_M-{str(m)[0]}"
        print(f"\n{'='*50}\nTraining Model: Boundary={b} | Conservation={c} | Monotonicity={m}\n{'='*50}")
        
        model = build_pinn_model(X_train_tsr.shape[1], flags)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
        
        # verbose=2 eklendi. Her epoch bir satır olarak yazdırılacak.
        model.fit(X_train_tsr, y_train_tsr, validation_split=0.1, epochs=CONFIG["EPOCHS"], batch_size=CONFIG["BATCH_SIZE"], callbacks=callbacks, verbose=2)
        
        # Test on OOD
        current_ood_df = ood_df.copy()
        current_ood_df['Predicted_Erc'] = model.predict(X_ood_tsr, batch_size=CONFIG["BATCH_SIZE"])[:, 0].flatten()
        current_ood_df['Residual'] = current_ood_df['Predicted_Erc'] - current_ood_df['True_Erc']
        
        mae_results = {'Config': config_name, 'Boundary': b, 'Conservation': c, 'Monotonicity': m}
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        colors = {'Extreme_Arid': 'red', 'High_Altitude': 'blue', 'Tropical_Monsoon': 'green'}
        
        for scenario in scenarios.keys():
            subset = current_ood_df[current_ood_df['Scenario'] == scenario]
            mae = mean_absolute_error(subset['True_Erc'], subset['Predicted_Erc'])
            mae_results[scenario + '_MAE'] = round(mae, 4)
            
            axes[0].scatter(subset['True_Erc'], subset['Predicted_Erc'], label=f"{scenario} (MAE: {mae:.2f})", color=colors[scenario], alpha=0.5, s=15)
            axes[1].scatter(subset['True_Erc'], subset['Residual'], label=scenario, color=colors[scenario], alpha=0.5, s=15)

        physics_title = f"PINN: Boundary={b}, Conservation={c}, Monotonicity={m}"
        axes[0].plot([current_ood_df['True_Erc'].min(), current_ood_df['True_Erc'].max()], [current_ood_df['True_Erc'].min(), current_ood_df['True_Erc'].max()], 'k--', lw=2)
        axes[0].set_title(f"True vs Predicted Erc\n{physics_title}")
        axes[0].grid(True); axes[0].legend()
        
        axes[1].axhline(0, color='black', linestyle='--', lw=2)
        axes[1].set_title(f"Residuals vs True Erc\n{physics_title}")
        axes[1].grid(True); axes[1].legend()

        fig_path = os.path.join(CONFIG["BASE_DIR"], f"pinn_results_{config_name}.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Grafik kaydedildi: {fig_path}")
        
        results_table.append(mae_results)

    # Karşılaştırma Tablosunu Oluşturma ve Yazdırma
    df_results = pd.DataFrame(results_table)
    csv_path = os.path.join(CONFIG["BASE_DIR"], "pinn_comparison_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    print("\n" + "="*80)
    print("FİZİKSEL KISITLAMALAR KARŞILAŞTIRMA TABLOSU")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    print(f"Tablo CSV olarak kaydedildi: {csv_path}")