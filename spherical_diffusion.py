# %% Import Python Libraries used for calculations.

import numpy as np
import sympy as sp

from matplotlib import rc
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
plt.rcParams.update({
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 16,
    'pdf.fonttype': 42,
    'font.family': 'Avenir',
    'font.size': 13,
    'xtick.direction': 'in',  # Set x-tick direction to 'in'
    'ytick.direction': 'in',  # Set y-tick direction to 'in'
    'xtick.major.size': 5,    # Set x-tick length
    'ytick.major.size': 5,    # Set y-tick length
    'xtick.major.pad': 6.5,   # Set x-tick padding
    'ytick.major.pad': 6.5    # Set y-tick padding
})

# %% 

# Define symbols
X_mf_init, X_FI_init = sp.symbols('X_mf_init X_FI_init')  # Initial mass fractions
S_mf_init, S_FI_init, Smass = sp.symbols('S_mf_init S_FI_init Smass')  # Initial compositions
S_mf_step, S_FI_step = sp.symbols('S_mf_step S_FI_step')  # Compositions at step
X_mf_step, X_FI_step, KD = sp.symbols('X_mf_step X_FI_step KD')  # End of step mass fraction and KD

# %%

# This says that the mass of the system is constant
mass_constant=sp.Eq(X_mf_init + X_FI_init, X_mf_step + X_FI_step)
# This writes a mass balance, but has subbsed in S_FI_step to S_mf_step
mass_balance=sp.Eq(X_mf_init * S_mf_init + X_FI_init * S_FI_init, X_mf_step * S_mf_step + X_FI_step * S_mf_step  * KD )
# Then we need to define how the system changes before and after
X_change=sp.Eq(X_mf_init-(S_mf_init*X_mf_init- S_mf_step*X_mf_step)/10**6, X_mf_step)

# Solve system of equations for S_mf_step, S_FI_step, and X_mf_step
solutions = sp.solve([mass_constant, mass_balance, X_change], (S_mf_step, X_mf_step, X_FI_step))

# Check the type of solutions
print("Solutions:")
print(solutions)

# If solutions are tuples, extract the S_mf_step component
if isinstance(solutions, list) and all(isinstance(sol, tuple) for sol in solutions):
    S_mf_step_solution = [sol[0] for sol in solutions]  # Extract first element (S_mf_step)
else:
    S_mf_step_solution = solutions[S_mf_step]  # Assume a dictionary output

print("S_mf_step solution:")
print(S_mf_step_solution)

# %%

# Function to get all roots for S_mf_step
def get_S_mf_step_roots(S_FI_init_val, S_mf_init_val, X_FI_init_val, X_mf_init_val, KD_val):
    # Define symbols
    S_FI_init, S_mf_init, X_FI_init, X_mf_init, KD = sp.symbols('S_FI_init S_mf_init X_FI_init X_mf_init KD')

    # Check if solutions are tuples
    if isinstance(solutions, list) and all(isinstance(sol, tuple) for sol in solutions):
        # Extract all S_mf_step solutions (first element of each tuple)
        S_mf_step_solution = [sol[0] for sol in solutions]  # First element of each tuple
    else:
        # For dictionary output (fallback if this happens)
        S_mf_step_solution = [solutions[S_mf_step]]
    
    # Create a dictionary to map input values to symbols
    subs_dict = {
        S_FI_init: S_FI_init_val,
        S_mf_init: S_mf_init_val,
        X_FI_init: X_FI_init_val,
        X_mf_init: X_mf_init_val,
        KD: KD_val
    }

    # Evaluate the solutions for S_mf_step
    evaluated_mf_steps = [sol.subs(subs_dict).evalf() for sol in S_mf_step_solution]
    
    return evaluated_mf_steps

# %%

# Assume `solutions` has been calculated using sp.solve()
mf_roots = get_S_mf_step_roots(S_FI_init_val=0, S_mf_init_val=1400, X_FI_init_val=0.5, X_mf_init_val=0.5, KD_val=70)

print("\nS_mf_step roots:")
for i, root in enumerate(mf_roots):
    print(f"Root {i + 1}: {root}")

# %%

def get_single_mf_root(S_FI_init_val, S_mf_init_val, X_FI_init_val, X_mf_init_val, KD_val):
    # Get all roots for S_mf_step
    mf_roots = get_S_mf_step_roots(S_FI_init_val, S_mf_init_val, X_FI_init_val, X_mf_init_val, KD_val)

    # Filter roots to those in the range (0, 1400)
    valid_roots = [root for root in mf_roots if 0 <= root <= 1400]

    # Check the number of valid roots
    if len(valid_roots) == 1:
        return valid_roots[0]  # Return the single valid root
    elif len(valid_roots) > 1:
        raise ValueError(f"Error: Multiple roots found in the range [0, 1400]: {valid_roots}")
    else:
        raise ValueError(f"Error: No valid root found in the range [0, 1400]: {mf_roots}")


# %%


test = get_single_mf_root(S_FI_init_val=0, S_mf_init_val=1400, X_FI_init_val=0.5, X_mf_init_val=0.5, KD_val=70)
test


# %% Solving two more equations. 


def second_solve_step(X_mf_init, X_FI_init, S_mf_init, S_mf_step, S_FI_init, S_FI_step):
    # Define symbols
    X_mf_step, X_FI_step = sp.symbols('X_mf_step X_FI_step')

    # Define the equations
    mass_constant = sp.Eq(X_mf_init + X_FI_init, X_mf_step + X_FI_step)
    X_change = sp.Eq(X_mf_init - (S_mf_init * X_mf_init - S_mf_step * X_mf_step) / 10**6, X_mf_step)

    # Solve the equations
    solutions = sp.solve([mass_constant, X_change], (X_mf_step, X_FI_step))

    return solutions

# %%


test2 = second_solve_step(X_mf_init=0.5, X_FI_init=0.5, S_mf_init=1400, S_mf_step=test, S_FI_init=0, S_FI_step=test*KD)
test2


# %%


def solve_one_mass_balance_step(S_FI_init_val, S_mf_init_val, X_FI_init_val, X_mf_init_val, KD_val):
    # first use Sympy to solve the value for S_mf, as this has two roots. 
    S_mf_step_calc=get_single_mf_root(S_FI_init_val, S_mf_init_val, X_FI_init_val, X_mf_init_val, KD_val)

    # Then calculate S in the FI after that
    S_FI_step_calc=S_mf_step_calc*KD
    # Then calculate the new X_FI and X_mf_step prior to diffusion
    results=second_solve_step(X_mf_init=X_mf_init_val, X_FI_init=X_FI_init_val, S_mf_init= S_mf_init_val, S_mf_step=S_mf_step_calc, S_FI_init=S_FI_init_val, S_FI_step=S_FI_step_calc)
    X_FI_step_calc=results[X_FI_step]
    X_mf_step_calc=results[X_mf_step]

    return S_mf_step_calc, S_FI_step_calc,  X_FI_step_calc, X_mf_step_calc


def solve_one_mass_balance_step_spatial(S_FI_init_val, S_mf_init_val, mass_FI, mass_melt, KD_val):
    # For the fluid inclusion, we assume S_FI = KD * S_mf.
    S_mf_step_calc = (mass_melt * S_mf_init_val + mass_FI * S_FI_init_val) / (mass_melt + mass_FI * KD_val)
    S_FI_step_calc = KD_val * S_mf_step_calc
    return S_mf_step_calc, S_FI_step_calc


# %% Let's set up some realistic geometry. 
## Lets assume a rectangle to help my brain, but the y axis disapears very quickly

# Radius of FI - but really width in this scenario
radius_FI_um=8
radius_FI=radius_FI_um/10**6 # to m

# Total radius of melt film (really width)
radius_melt_um=1
radius_melt=radius_melt_um/10**6 # to m

density_melt=2700
density_FI=160

# Calculate the mass of each in 3D this time
constant = 4/3 *np.pi
vol_FI=constant * radius_FI**3
vol_melt=constant * (radius_melt+radius_FI)**3 - vol_FI


mass_melt_all=density_melt*vol_melt
mass_FI_all=density_melt*vol_FI
mass_melt=mass_melt_all/(mass_melt_all+mass_FI_all)
mass_FI=mass_FI_all/(mass_melt_all+mass_FI_all)
print('Mass % FI')
print(100*mass_FI/(mass_melt + mass_FI))

# KD value
KD=100

# Initial S in the melt
S_melt_init=1400
S_FI_init=0

# %% Solve the equations prior to diffusion

S_mf_step_calc, S_FI_step_calc, X_FI_step_calc, X_mf_step_calc = solve_one_mass_balance_step(S_FI_init_val=S_FI_init, S_mf_init_val=S_melt_init, 
                                                                                             X_FI_init_val=mass_FI, X_mf_init_val=mass_melt, KD_val=KD)

print('out of the function')
print('S_mf_step_calc (ppm):', S_mf_step_calc)

print('S_FI_step_calc (ppm):', S_FI_step_calc)

print('X_FI_step_calc:', X_FI_step_calc)

print('X_mf_step_calc:', X_mf_step_calc)


# %% 

def convert_s_ppm_to_so2_molperc(S_ppm):
    """
    Convert sulfur concentration (ppm) to SO2 mol% for scalar or array input.

    Parameters:
        S_ppm: Scalar or numpy array of sulfur concentrations in ppm.
    Returns:
        mol_frac_SO2: Scalar or numpy array of SO2 mol% values.
    """
    # Ensure input is a NumPy array for element-wise operations
    S_ppm = np.asarray(S_ppm)
    
    # Convert ppm to weight percent
    S_wt = S_ppm / 10**4
    # Calculate moles of S
    S_moles = S_wt / 32.065
    # Moles of O from SO2
    O_moles_from_SO2 = S_moles * 2
    # Oxygen weight contribution from SO2
    O_wt_from_SO2 = O_moles_from_SO2 * 16
    # Remaining mass as CO2
    mass_CO2 = 100 - S_wt - O_wt_from_SO2
    # Moles of CO2
    moles_CO2 = mass_CO2 / 44.009
    # Mole fraction of SO2
    mol_frac_SO2 = S_moles / (S_moles + moles_CO2)
    
    return mol_frac_SO2

# %% 

def run_diffusion_spherical(D, radius_melt, radius_melt_film, radius_fluid_inclusion,
                            S_melt_init, S_FI_init, mass_FI, mass_melt, 
                            KD, total_time=600, save_interval=1):
    """
    Simulate sulfur diffusion in spherical coordinates with a fluid inclusion and mass balance.

    Parameters:
        D (float): Diffusivity (m^2/s).
        radius_melt (float): Radius of the melt (m).
        radius_melt_film (float): Radius of the melt film (m).
        radius_fluid_inclusion (float): Radius of the spherical fluid inclusion (m).
        S_melt_init (float): Initial sulfur concentration in the melt / melt film (ppm).
        S_FI_init (float): Initial sulfur concentration in the fluid inclusion (ppm).
        mass_FI (float): Mass of the fluid inclusion (kg).
        mass_melt (float): Mass of the melt film (kg).
        KD (float): Partition coefficient.
        total_time (float): Total simulation time in seconds (default: 600s).
    """

    ### Setup grid and initial parameters
    dr = radius_melt_film   # Radial step size
    total_radius = (radius_melt + radius_melt_film + radius_fluid_inclusion)
    r = np.arange(0, total_radius + dr, dr)  # Radial grid
    Nr = len(r)  # Number of radial points
    dt = (0.25 * dr**2) / D  # Time step size based on CFL condition
    time_steps = int(total_time / dt)  # Number of time steps
    save_interval = save_interval  # Frequency of saving the results

    # Initialize concentration array
    C = np.full(Nr, S_melt_init, dtype=float)  # Initial concentration in melt

    ### Initial mass balance
    S_mf_initial, S_FI_initial, X_FI_initial, X_mf_initial = solve_one_mass_balance_step(
        S_FI_init_val=S_FI_init, 
        S_mf_init_val=S_melt_init,
        X_FI_init_val=mass_FI, 
        X_mf_init_val=mass_melt,
        KD_val=KD
    )

    print('Initial:', round(S_mf_initial, 4), round(S_FI_initial, 4), X_FI_initial, X_mf_initial)

    ### Define masks for geometry
    fluid_inclusion_mask = r <= radius_fluid_inclusion
    melt_film_mask = (r > radius_fluid_inclusion) & (r <= radius_fluid_inclusion + radius_melt_film)
    melt_mask = r > radius_fluid_inclusion + radius_melt_film
    interface_index = np.argmax(r > radius_fluid_inclusion)

    C[r <= radius_fluid_inclusion] = S_FI_initial  # Initial concentration in fluid inclusion
    C[melt_film_mask] = S_mf_initial  # Melt film is set to the mass balance value (~65.94 ppm)

    ### Initialize storage for outputs
    C_profile = np.zeros((time_steps // save_interval + 1, Nr))
    C_profile[0, :] = C
    S_FI_over_time = np.zeros(time_steps)
    S_mf_over_time = np.zeros(time_steps)

    ### Diffusion and mass balance loop
    for t in range(time_steps):
        try:
            # Step 4.1: Update diffusion in spherical coordinates
            C_new = C.copy()
            for i in range(1, Nr - 1):  # Skip boundaries
                if i == interface_index:
                    # The grid cell i is the first point in the melt film
                    # Its neighbor at i-1 is in the fluid inclusion.
                    # Impose a zero gradient by setting C[i-1] equal to C[i] in the finite-difference.
                    # That is, replace the backward difference by zero.
                    # One way is to use a one-sided forward difference:
                    dC_dr = (C[i+1] - C[i]) / dr
                    # For the second derivative, assume:
                    d2C_dr2 = (C[i+1] - C[i]) / dr**2
                else:
                    # For all other points use the central difference.
                    dC_dr = (C[i+1] - C[i-1]) / (2*dr)
                    d2C_dr2 = (C[i+1] - 2 * C[i] + C[i-1]) / dr**2

                # Update concentration using the spherical Laplacian:
                C_new[i] = C[i] + D * dt * (d2C_dr2 + (2 / r[i]) * dC_dr)

            # Apply boundary conditions
            C_new[0] = C_new[1]  # Symmetry condition at r = 0
            C_new[-1] = C_new[-2]  # No-flux condition at outer boundary
            # interface_index = np.argmax(r >= radius_fluid_inclusion) # At fluid inclusion and melt film interface
            # C_new[interface_index] = C_new[interface_index + 1]  # No-flux condition at interface
            C = C_new

            # Compute average sulfur concentration in the melt film
            S_mf_diffusion = np.mean(C[melt_film_mask])  # Average S in melt film
            # S_mf_min = np.nanmin(C[melt_film_mask])  # Average S in melt film
            # S_mf_max = np.nanmax(C[melt_film_mask])  # Average S in melt film
            print(round(t * dt, 4), 'seconds, melt film S :', round(S_mf_diffusion, 4))
            # 'min', round(S_mf_min, 4), 'max', round(S_mf_max, 4))

            # Compute change in melt mass
            deltaS = S_mf_diffusion - S_melt_init
            X_mf_step_new = mass_melt + mass_melt * deltaS / 10**6

            # Solve new mass balance equation
            S_mf_new, S_FI_step_calc, X_FI_step_calc, X_mf_step_calc = solve_one_mass_balance_step(
                S_FI_init_val=S_FI_init,
                S_mf_init_val=S_mf_diffusion,
                X_FI_init_val=mass_FI,
                X_mf_init_val=X_mf_step_new,
                KD_val=KD
            )
            print(round(S_mf_new, 4), round(S_FI_step_calc), X_FI_step_calc, X_mf_step_calc)

            # Update sulfur concentrations
            S_FI_over_time[t] = S_FI_step_calc  # Track sulfur in the fluid inclusion
            S_mf_over_time[t] = S_mf_diffusion  # Track sulfur in the melt film
            C[fluid_inclusion_mask] = S_FI_step_calc  # Update fluid inclusion concentration

            # Save the concentration profile at intervals
            if t % save_interval == 0:
                C_profile[t // save_interval, :] = C

        except ValueError as e:
            # Save partial results and break
            print(f"Error at time {t * dt} seconds: {e}")
            break

    ### Return results even if terminated early
    return C_profile[:t // save_interval + 1], S_FI_over_time[:t], S_mf_over_time[:t], dr, dt, save_interval


# %% 

# Parameters
D = 1e-12  # Diffusivity (m^2/s)
radius_melt = 50*1e-6  # Radius of the melt (m)
radius_melt_film = 0.1*1e-6  # Radius of the melt film (m)
radius_fluid_inclusion = 8*1e-6  # Radius of the fluid inclusion (m)
total_radius = radius_melt + radius_melt_film + radius_fluid_inclusion

S_melt_init = 1400  # Sulfur concentration in the melt (ppm)
S_FI_init = 0 # Sulfur concentration in the fluid inclusion (ppm)

density_melt = 2700 # kg/m^3
density_FI = 230 # kg/m^3
KD = 100 

# Mass of the fluid inclusion
volume_fluid_inclusion = (4/3) * np.pi * radius_fluid_inclusion**3
mass_FI = density_FI * volume_fluid_inclusion

# Total melt volume (outer sphere minus inner sphere)
radius_outer_melt = radius_fluid_inclusion + radius_melt_film
volume_total_melt = ((4/3) * np.pi * (radius_outer_melt**3)) - volume_fluid_inclusion
mass_melt = density_melt * volume_total_melt

C_profile_010, S_FI_over_time_010, S_mf_over_time_010, dr_010, dt_010, save_interval_010 = run_diffusion_spherical(D, radius_melt, radius_melt_film, radius_fluid_inclusion,
                                                                                           S_melt_init, S_FI_init, mass_FI, mass_melt, 
                                                                                           KD, total_time=30)

# %%

np.savez('results/spherical_C_profile_010.npz', C_profile=C_profile_010, S_FI_over_time=S_FI_over_time_010, S_mf_over_time=S_mf_over_time_010, dr=dr_010, dt=dt_010, save_interval=save_interval_010)
np.savez('results/spherical_S_results_010.npz', S_FI_over_time=S_FI_over_time_010, S_mf_over_time=S_mf_over_time_010, dr=dr_010, dt=dt_010, save_interval=save_interval_010)

# %%

from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

def animate_diffusion_full_cross_section(C_profile, C_profile_molperc, r, dt, 
                                          radius_fluid_inclusion,
                                          save_path='results/diffusion_full_cross_section.gif',
                                          frame_step=1):
    """
    Create an animation of sulfur diffusion with a full 2D cross-section (XZ-plane)
    and a 1D transect. The two subplots will be of equal size, and the two colorbars 
    on the 2D subplot will span the same vertical height as the 2D axes.
    
    Parameters:
        C_profile (numpy.ndarray): Concentration profile over time (ppm).
        C_profile_molperc (numpy.ndarray): Concentration profile over time in mol% for the fluid inclusion.
        r (numpy.ndarray): Radial grid (1D array).
        dt (float): Time step size.
        radius_fluid_inclusion (float): Radius of the fluid inclusion.
        save_path (str): Path to save the animation as a GIF.
        frame_step (int): Subsampling rate for the time frames.
    """

    # Subsample the time steps
    frame_step = frame_step
    frames = range(0, C_profile.shape[0], frame_step)

    # Create a figure with two equally sized subplots using GridSpec.
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1])
    gs.update(wspace=0.4)  # Increase horizontal spacing between subplots
    
    # Left subplot: 1D radial transect.
    ax1 = fig.add_subplot(gs[0])
    # Right subplot: 2D cross-section.
    ax2 = fig.add_subplot(gs[1])
    
    # -----------------------------
    # 1D Radial Transect Setup
    # -----------------------------
    # Create a twin y-axis so that we can plot two different datasets.
    ax1_twin = ax1.twinx()
    
    # Mask the fluid inclusion region in C_profile with NaN
    fluid_inclusion_mask = r <= radius_fluid_inclusion  # Mask for the fluid inclusion region
    C_profile_masked = np.copy(C_profile)
    C_profile_masked[:, fluid_inclusion_mask] = np.nan  # Mask fluid inclusion region

    # Mask the melt region in C_profile_molperc with NaN
    C_profile_molperc_masked = np.copy(C_profile_molperc)
    C_profile_molperc_masked[:, ~fluid_inclusion_mask] = np.nan  # Mask melt region

    # Plot the full profile (red line).
    line1, = ax1.plot(r * 1e6, C_profile[0, :], label='S (ppm)', color='#E42211')

    # Plot the fluid inclusion profile (blue dashed line) on the twin axis.
    line2, = ax1_twin.plot(r * 1e6, C_profile_molperc_masked[0, :], label='$\\mathregular{{S_{{FI}}}}$ (mol%)', 
                           color='#0A306B', linestyle='--', linewidth=3)
    
    ax1.set_xlabel('Radius (µm)')
    ax1.set_ylabel('S (ppm)', color='#E42211')
    ax1_twin.set_ylabel('$\\mathregular{{S_{{FI}}}}$ (mol%)', color='#0A306B')
    ax1.set_title('1D Radial Transect')
    ax1.set_xlim([0, r[-1]*1e6])
    ax1.set_ylim([0, np.nanmax(C_profile)*1.125])
    ax1_twin.set_ylim([0, np.nanmax(C_profile_molperc_masked)*1.125])
    ax1.grid()
    
    # Add legends; you may adjust the locations as needed.
    ax1.legend(loc='lower center', fontsize=10)
    ax1_twin.legend(loc='lower right', fontsize=10)
    
    # -----------------------------
    # 2D Cross-Section Setup (XZ-plane)
    # -----------------------------
    # Create a grid for the 2D plot.
    x = np.linspace(-r[-1], r[-1], 200)
    z = np.linspace(-r[-1], r[-1], 200)
    X, Z = np.meshgrid(x, z)
    R = np.sqrt(X**2 + Z**2)
    
    # Map the 1D concentration profile (melt, ppm) to the 2D grid.
    C_2D = np.interp(R.flatten(), r, C_profile_masked[0, :]).reshape(X.shape)
    # Map the fluid inclusion (mol%) data to the 2D grid.
    C_FI_2D = np.interp(R.flatten(), r, C_profile_molperc_masked[0, :]).reshape(X.shape)
    
    # Plot the melt concentration using imshow.
    im1 = ax2.imshow(C_2D, 
                     extent=(-r[-1]*1e6, r[-1]*1e6, -r[-1]*1e6, r[-1]*1e6),
                     origin='lower', cmap='Reds', 
                     vmin=0, vmax=np.nanmax(C_profile_masked))

    # Use an axes divider to create colorbar axes that match the vertical height of ax2.
    divider = make_axes_locatable(ax2)
    # Create the first colorbar axis (for the melt data) to the right of ax2.
    cax1 = divider.append_axes("right", size="5%", pad=0.075)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.ax.tick_params(labelsize=10, pad=3)  # set tick label size for the first colorbar
    cbar1.ax.set_title('$\\mathregular{{S_{{melt}}}}$\n(ppm)', fontsize=10)
    
    # Plot the fluid inclusion concentration overlay.
    im2 = ax2.imshow(C_FI_2D, 
                     extent=(-r[-1]*1e6, r[-1]*1e6, -r[-1]*1e6, r[-1]*1e6),
                     origin='lower', cmap='Blues',
                     vmin=0, vmax=np.nanmax(C_profile_molperc_masked))
    # Create a second colorbar axis for the fluid inclusion data.
    cax2 = divider.append_axes("right", size="5%", pad=0.45)
    cbar2 = plt.colorbar(im2, cax=cax2)
    cbar2.ax.tick_params(labelsize=10, pad=3)  # set tick label size for the first colorbar
    cbar2.ax.set_title('$\\mathregular{{S_{{FI}}}}$\n(mol%)', fontsize=10)
    
    ax2.set_title('2D Cross-Section')
    ax2.set_xlabel('X-Y-Z Axis (µm)')
    ax2.set_ylabel('X-Y-Z Axis (µm)')
    
    # -----------------------------
    # Update function for the animation
    # -----------------------------
    def update(frame):
        # Update 1D transect.
        line1.set_ydata(C_profile[frame, :])
        line2.set_ydata(C_profile_molperc_masked[frame, :])
        
        # Update the 2D images.
        C_2D = np.interp(R.flatten(), r, C_profile_masked[frame, :]).reshape(X.shape)
        im1.set_data(C_2D)
        C_FI_2D = np.interp(R.flatten(), r, C_profile_molperc_masked[frame, :]).reshape(X.shape)
        im2.set_data(C_FI_2D)
        
        # Update titles with current time.
        ax1.set_title(f'1D Radial Transect, t={frame * dt * save_interval:.3f}s')
        ax2.set_title(f'2D Cross-Section, $\\mathregular{{S_{{FI}}}}$={np.round(np.nanmax(C_profile_molperc_masked[frame, :]), 4)} mol%')
        return line1, line2, im1, im2
    
    # Create the animation.
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save the animation.
    anim.save(save_path, writer='pillow', fps=10)
    print(f'Animation saved to {save_path}')
    
    plt.show()

# %% 

C_profile_molperc = np.copy(C_profile_15)  # Create a copy of C_profile
r = np.arange(0, total_radius + dr_15, dr_15)  # Radial grid
dr = radius_melt_film # / 20
fluid_inclusion_mask = r <= radius_fluid_inclusion  # Mask for the fluid inclusion region
C_profile_molperc[:, fluid_inclusion_mask] = convert_s_ppm_to_so2_molperc(C_profile_15[:, fluid_inclusion_mask])
save_interval=5

# animate_diffusion_full_cross_section(C_profile, r, dt, save_path='diffusion_full_cross_section.gif')
animate_diffusion_full_cross_section(C_profile_15, C_profile_molperc, r, dt_15, radius_fluid_inclusion, 
                                     save_path='results/diffusion_full_cross_section_test.gif')

# %% 

# Create a time axis (in seconds) that matches S_FI_over_time.

results_15 = np.load('results/spherical_S_results_15.npz')
time_array_15 = np.arange(len(results_15['S_FI_over_time'])) * results_15['dt']
S_FI_over_time_molperc_15 = convert_s_ppm_to_so2_molperc(results_15['S_FI_over_time'])

results_1 = np.load('results/spherical_S_results_1.npz')
time_array_1 = np.arange(len(results_1['S_FI_over_time'])) * results_1['dt']
S_FI_over_time_molperc_1 = convert_s_ppm_to_so2_molperc(results_1['S_FI_over_time'])

results_050 = np.load('results/spherical_S_results.npz')
time_array_050 = np.arange(len(results_050['S_FI_over_time'])) * results_050['dt']
S_FI_over_time_molperc_050 = convert_s_ppm_to_so2_molperc(results_050['S_FI_over_time'])

results_005 = np.load('results/spherical_S_results.npz')
time_array_005 = np.arange(len(results_005['S_FI_over_time'])) * results_005['dt']
S_FI_over_time_molperc_005 = convert_s_ppm_to_so2_molperc(results_005['S_FI_over_time'])

results_003 = np.load('results/spherical_S_results.npz')
time_array_003 = np.arange(len(results_003['S_FI_over_time'])) * results_003['dt']
S_FI_over_time_molperc_003 = convert_s_ppm_to_so2_molperc(results_003['S_FI_over_time'])

results_001 = np.load('results/spherical_S_results_001.npz')
time_array_001 = np.arange(len(results_001['S_FI_over_time'])) * results_001['dt']
S_FI_over_time_molperc_001 = convert_s_ppm_to_so2_molperc(results_001['S_FI_over_time'])


fig, ax2 = plt.subplots(figsize=(8, 6))
# Primary y-axis for S_mf (ppm)
# ax1.plot(time_array, S_mf_over_time, lw=2, ls='--', label='Sulfur in Melt Film (ppm), dx=1', color='#E42211')
# ax1.plot(time_array_smalldx, results_005['S_mf_over_time'], lw=2, label='Sulfur in Melt Film (ppm), dx<1', color='#E42211')
# ax1.set_xlabel('Time (s)')
# ax1.set_ylabel('$\\mathregular{{S_{{melt film}}}}$ (ppm)', color='#E42211')
# ax1.tick_params(axis='y', labelcolor='#E42211')
# ax1.legend(fontsize=10, loc='lower center')

# Secondary y-axis for S_FI (mol%)
# ax2 = ax1.twinx()
# ax2.plot(time_array_15, S_FI_over_time_molperc_15, lw=2, ls='--', label='Sulfur in FI, dx=1.5 (mol%)', color='grey') #, alpha=0.5)
# ax2.plot(time_array_1, S_FI_over_time_molperc_1, lw=2, ls='--', label='Sulfur in FI, dx=1 (mol%)', color='#0A306B') #, alpha=0.5)
# ax2.plot(time_array_050, S_FI_over_time_molperc_050, lw=2, label='Sulfur in FI, dx=0.050 (mol%)', color='magenta')
ax2.plot(time_array_005, results_005['S_FI_over_time']/results_005['S_mf_over_time'], lw=2, label='Sulfur in FI, dx=0.005 (mol%)', color='k')
# ax2.plot(time_array_003, S_FI_over_time_molperc_003, lw=2, ls='--', label='Sulfur in FI, dx=0.003 (mol%)', color='cyan')
# ax2.plot(time_array_001, S_FI_over_time_molperc_001, lw=2, ls='--', label='Sulfur in FI, dx=0.001 (mol%)', color='red')
ax2.set_ylabel('$\\mathregular{{S_{{FI}}}}$ (mol%)', color='#0A306B')
ax2.tick_params(axis='y', labelcolor='#0A306B')
ax2.legend(fontsize=10)
fig.tight_layout()
plt.savefig('results/diffusion_over_time_dx.pdf')
plt.show()

# %%