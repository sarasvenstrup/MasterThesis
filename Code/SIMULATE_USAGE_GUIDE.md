"""
REFACTORED SIMULATE_MODEL.PY - USAGE GUIDE
===========================================

The simulate_model.py script has been refactored into modular functions that can be called 
independently, similar to the Training.py structure.

KEY CHANGES:
============

1. REMOVED: All code was in main()
2. ADDED: Separate modular functions for each stage:

   Stage 1: load_and_setup_model()
   Stage 2: compute_latent_statistics()
   Stage 3: run_simulation()
   Stage 4: analyze_paths()
   Stage 5: decode_and_save_results()
   Stage 6: generate_plots()

USAGE EXAMPLES:
===============

1. RUN THE FULL SIMULATION (from command line):
   -----------------------------------------------
   python Code/simulate_model.py --n_paths 100 --n_steps 120

2. RUN STEP-BY-STEP (from Python console):
   -----------------------------------------------
   
   # Load model and setup
   from Code.simulate_model import *
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = load_and_setup_model(device, use="bbg", latent_dim=2, epochs=100)
   
   # Load data and compute diagnostics
   meta, X_tensor, _, _, tenors, _, _, _ = my_data(use="bbg")
   X_tensor = X_tensor.double()
   z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(
       model, X_tensor, device, latent_dim=2
   )
   
   # Load initial curve and encode it
   S0, meta_row, X_tensor, meta = load_initial_curve("bbg", -1, device)
   with torch.no_grad():
       z0 = model.encoder(S0)
   
   # Run simulation with different parameters
   z_paths, r_paths, mu_paths, L_paths = run_simulation(
       model, z0, n_paths=50, n_steps=60, dt=1/12, device=device,
       latent_dim=2, simple_diffusion=False, kappa=0.5, theta=0.0,
       sigma_simple=0.1, discretization="euler"
   )
   
   # Analyze paths
   analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim=2)
   
   # Decode and save results
   swap_df, latent_df, out_dir, times, early_stop_time = decode_and_save_results(
       model, z_paths, r_paths, z_train_mean, z_train_cov, device,
       n_steps=60, n_paths=50, dt=1/12, tenors=tenors,
       use="bbg", latent_dim=2, epochs=100
   )
   
   # Generate plots
   generate_plots(z_paths, r_paths, mu_paths, L_paths, swap_df, tenors,
                  out_dir, times, n_paths=50, n_steps=60,
                  use="bbg", latent_dim=2, epochs=100, dt=1/12)


3. RUN ONLY SPECIFIC STAGES (from Python):
   -----------------------------------------------
   
   # If you already have z_paths and want to decode without re-simulating:
   swap_df, latent_df, out_dir, times, _ = decode_and_save_results(...)
   generate_plots(...)
   
   # If you want to try different plot settings:
   generate_plots(..., n_steps=120)  # Re-plot with different parameters


FUNCTION SIGNATURES:
====================

load_and_setup_model(device, use, latent_dim, epochs)
  - Load checkpoint and verify variant consistency
  - Returns: model (FullModel)

compute_latent_statistics(model, X_tensor, device, latent_dim)
  - Compute training latent region statistics
  - Returns: z_train_mean, z_train_cov, z_train_std

run_simulation(model, z0, n_paths, n_steps, dt, device, latent_dim, 
               simple_diffusion, kappa, theta, sigma_simple, discretization)
  - Run latent path simulation
  - Returns: z_paths, r_paths, mu_paths, L_paths

analyze_paths(z_paths, r_paths, mu_paths, L_paths, latent_dim)
  - Analyze and print diagnostics for simulated paths
  - Returns: None (prints diagnostics)

decode_and_save_results(model, z_paths, r_paths, z_train_mean, z_train_cov, 
                        device, n_steps, n_paths, dt, tenors, use, latent_dim, epochs)
  - Decode latent paths to swap curves and save results
  - Returns: swap_df, latent_df, out_dir, times, early_stop_time

generate_plots(z_paths, r_paths, mu_paths, L_paths, swap_df, tenors, out_dir, 
               times, n_paths, n_steps, use, latent_dim, epochs, dt)
  - Generate and save all plots
  - Returns: None (saves plots to out_dir)


MODULAR WORKFLOW EXAMPLE:
=========================

# Scenario: Run simulation with different configurations without reloading model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_and_setup_model(device, use="bbg", latent_dim=2, epochs=100)

# Load data once
meta, X_tensor, _, _, tenors, _, _, _ = my_data(use="bbg")
X_tensor = X_tensor.double()
z_train_mean, z_train_cov, z_train_std = compute_latent_statistics(model, X_tensor, device, 2)

S0, _, X_tensor, meta = load_initial_curve("bbg", -1, device)
with torch.no_grad():
    z0 = model.encoder(S0)

# Experiment 1: Quick test with fewer paths
print("Experiment 1: 50 paths, 60 steps")
z1, r1, mu1, L1 = run_simulation(model, z0, 50, 60, 1/12, device, 2, False, 0.5, 0.0, 0.1, "euler")
analyze_paths(z1, r1, mu1, L1, 2)
df_swap1, df_lat1, _, _, _ = decode_and_save_results(model, z1, r1, z_train_mean, z_train_cov, device, 60, 50, 1/12, tenors, "bbg", 2, 100)
generate_plots(z1, r1, mu1, L1, df_swap1, tenors, "Figures/simulations", np.arange(61)*1/12, 50, 60, "bbg", 2, 100, 1/12)

# Experiment 2: Full simulation with more paths
print("Experiment 2: 200 paths, 120 steps")
z2, r2, mu2, L2 = run_simulation(model, z0, 200, 120, 1/12, device, 2, False, 0.5, 0.0, 0.1, "euler")
analyze_paths(z2, r2, mu2, L2, 2)
df_swap2, df_lat2, _, _, _ = decode_and_save_results(model, z2, r2, z_train_mean, z_train_cov, device, 120, 200, 1/12, tenors, "bbg", 2, 100)
generate_plots(z2, r2, mu2, L2, df_swap2, tenors, "Figures/simulations", np.arange(121)*1/12, 200, 120, "bbg", 2, 100, 1/12)


KEY BENEFITS:
=============

✓ Modularity: Each stage can be called independently
✓ Debugging: Easy to test individual components
✓ Experimentation: Run multiple simulations with same model
✓ Flexibility: Skip stages as needed
✓ Reusability: Functions can be imported and used in other scripts
✓ Maintainability: Similar structure to Training.py
"""

