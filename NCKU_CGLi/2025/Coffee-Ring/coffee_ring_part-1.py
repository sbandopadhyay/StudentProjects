"""
================================================================================
DEPTH-AVERAGED COFFEE RING SIMULATION
================================================================================

Mathematical Model (Verified):
------------------------------
1. Geometry:        h(r,t) = (R^2 - r^2)theta(t) / (2R)
2. Contact angle:   theta(t) = theta_0(1 - t/t_f)
3. Height rate:     dh/dt = -h / (t_f - t)
4. Evaporation:     J(r) = J_0(1 - r_tilde^2)^(-lambda),  lambda = (pi - 2theta)/(2pi - 2theta)
5. Velocity:        u_bar_r = (1/rh) int_0^r [h'/(t_f-t) - J'/rho] r' dr'
6. Concentration:   dc/dt + u_bar_r dc/dr = (D/rh)d_r(rh d_r c) + cJ/(rhoh) - S/h
7. Deposition:      dsigma/dt = k_dep * c * exp(-h/h*)

References:
-----------
- Deegan et al. (1997) Nature - Coffee ring discovery
- Hu & Larson (2002) J. Phys. Chem. B - Evaporation flux
- Popov (2005) Phys. Rev. E - Analytical theory

================================================================================
"""

import numpy as np
from numpy import pi, sqrt, exp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass


@dataclass
class Params:
    """Physical and numerical parameters"""
    # Geometry
    R: float = 1.0              # Contact radius [mm]
    theta0: float = 0.5         # Initial contact angle [rad] (~28.6deg)
    
    # Fluid
    rho: float = 1.0            # Density (normalized)
    D: float = 0.001            # Particle diffusivity
    
    # Evaporation
    J0: float = 0.1             # Evaporation prefactor
    RH: float = 0.5             # Relative humidity (for info)
    
    # Deposition
    k_dep: float = 0.1          # Deposition rate constant
    h_star: float = 0.02        # Characteristic height for deposition
    
    # Numerics
    Nr: int = 200               # Radial grid points
    eps: float = 0.02           # Singularity regularization
    dt_safety: float = 0.5      # CFL safety factor
    
    # Simulation
    t_end_frac: float = 0.95    # End at 95% of drying time
    
    # Initial particle distribution
    # Options: 'uniform', 'gaussian', 'ring', 'random', 'center', 'custom'
    init_type: str = 'uniform'
    init_center: float = 0.0    # Center of Gaussian/ring (r/R)
    init_width: float = 0.2     # Width of Gaussian/ring
    init_seed: int = None       # Random seed for reproducibility


class CoffeeRingSim:
    """
    Eulerian depth-averaged coffee ring simulator.
    
    Solves coupled equations for:
    - h(r,t): film height
    - c(r,t): depth-averaged particle concentration  
    - sigma(r,t): deposited particle density
    """
    
    def __init__(self, params=None):
        self.p = params or Params()
        self._setup()
        
    def _setup(self):
        """Initialize grid and fields"""
        p = self.p
        
        # Radial grid (avoid r=0 singularity)
        self.r = np.linspace(p.eps * p.R, p.R, p.Nr)
        self.dr = self.r[1] - self.r[0]
        self.r_norm = self.r / p.R  # r_tilde = r/R
        
        # Initial contact angle
        self.theta = p.theta0
        
        # Initial height profile
        self.h = self._calc_height()
        
        # Initial volume
        V0 = np.trapz(2 * pi * self.r * self.h, self.r)
        
        # Evaporation flux
        self.J = self._calc_evap()
        
        # Total evaporation rate: dV/dt = int J/rho * 2pir dr
        dVdt = np.trapz(2 * pi * self.r * self.J / p.rho, self.r)
        
        # Consistent drying time: t_f = V0 / (dV/dt)
        self.t_f = V0 / dVdt
        
        # Time
        self.t = 0.0
        self.t_end = p.t_end_frac * self.t_f
        
        # Fields
        self.u = np.zeros(p.Nr)
        self.c = self._init_concentration()   # Initial concentration
        self.sigma = np.zeros(p.Nr)           # Deposited amount
        
        # Diagnostics
        self.mass0 = self._total_mass()
    
    def _init_concentration(self):
        """
        Initialize particle concentration based on init_type.
        
        Options:
        - 'uniform': c = 1 everywhere (default)
        - 'gaussian': Gaussian centered at init_center with init_width
        - 'ring': Ring distribution at init_center
        - 'center': Concentrated at center
        - 'random': Random fluctuations around 1
        - 'custom': User provides via set_concentration()
        """
        p = self.p
        r_norm = self.r / p.R
        
        if p.init_seed is not None:
            np.random.seed(p.init_seed)
        
        if p.init_type == 'uniform':
            c = np.ones(p.Nr)
            
        elif p.init_type == 'gaussian':
            # Gaussian centered at init_center
            c = np.exp(-((r_norm - p.init_center)**2) / (2 * p.init_width**2))
            c = np.maximum(c, 0.01)  # Small background
            
        elif p.init_type == 'ring':
            # Ring at init_center
            c = np.exp(-((r_norm - p.init_center)**2) / (2 * p.init_width**2))
            c = np.maximum(c, 0.01)
            
        elif p.init_type == 'center':
            # Concentrated at center (Gaussian at r=0)
            c = np.exp(-(r_norm**2) / (2 * p.init_width**2))
            c = np.maximum(c, 0.01)
            
        elif p.init_type == 'random':
            # Random fluctuations: 1 + noise
            noise = np.random.randn(p.Nr) * 0.3
            # Smooth the noise
            from scipy.ndimage import gaussian_filter1d
            try:
                noise = gaussian_filter1d(noise, sigma=5)
            except:
                # Fallback if scipy not available
                kernel = np.exp(-np.arange(-10, 11)**2 / 20)
                kernel /= kernel.sum()
                noise = np.convolve(noise, kernel, mode='same')
            c = 1.0 + noise
            c = np.maximum(c, 0.1)
            
        elif p.init_type == 'blobs':
            # Multiple random blobs
            n_blobs = np.random.randint(3, 7)
            c = 0.1 * np.ones(p.Nr)
            for _ in range(n_blobs):
                r_blob = np.random.uniform(0.1, 0.8)
                w_blob = np.random.uniform(0.05, 0.15)
                amp = np.random.uniform(0.5, 2.0)
                c += amp * np.exp(-((r_norm - r_blob)**2) / (2 * w_blob**2))
        
        else:  # 'custom' or unknown
            c = np.ones(p.Nr)
        
        # Normalize so total mass = 1 (optional)
        # mass = np.trapz(2 * pi * self.r * self.h * c, self.r)
        # c /= mass
        
        return c
    
    def set_concentration(self, c_array):
        """Set custom initial concentration profile"""
        if len(c_array) == self.p.Nr:
            self.c = np.array(c_array)
            self.mass0 = self._total_mass()
        else:
            raise ValueError(f"Array length {len(c_array)} != Nr {self.p.Nr}")
        
    def _calc_height(self):
        """h(r) = (R^2 - r^2)theta / (2R)"""
        p = self.p
        h = (p.R**2 - self.r**2) * self.theta / (2 * p.R)
        return np.maximum(h, 1e-10)
    
    def _calc_evap(self):
        """
        J(r) = J_0 (1 - r_tilde^2)^(-lambda)
        lambda = (pi - 2theta) / (2pi - 2theta)
        """
        p = self.p
        
        # Singularity exponent
        lam = (pi - 2*self.theta) / (2*pi - 2*self.theta)
        
        # Regularize near edge
        arg = 1 - self.r_norm**2
        arg = np.maximum(arg, p.eps)
        
        J = p.J0 * arg**(-lam)
        return J
    
    def _calc_velocity(self):
        """
        u_bar_r = (1/rh) int_0^r [h'/(t_f-t) - J'/rho] r' dr'
        
        Uses trapezoidal integration with edge regularization.
        """
        p = self.p
        tau = max(self.t_f - self.t, 0.01 * self.t_f)
        
        # Integrand: [h/(t_f-t) - J/rho] * r
        integrand = (self.h / tau - self.J / p.rho) * self.r
        
        # Cumulative integral (trapezoidal)
        integral = np.zeros(p.Nr)
        for i in range(1, p.Nr):
            integral[i] = integral[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * self.dr
        
        # u_bar_r = integral / (r * h) with regularization
        h_reg = np.maximum(self.h, 0.01 * self.h[0])  # Prevent h->0
        self.u = integral / (self.r * h_reg)
        self.u[0] = 0  # Symmetry at center
        
        # Clip to physical range (outward, bounded)
        u_scale = p.R / self.t_f
        self.u = np.clip(self.u, -u_scale, 50 * u_scale)
        
        # Smooth near edge to avoid oscillations
        self.u[-5:] = np.linspace(self.u[-6], self.u[-6] * 1.5, 5)
        
    def _calc_dhdt(self):
        """dh/dt = -h / (t_f - t)"""
        tau = max(self.t_f - self.t, 1e-10)
        return -self.h / tau
    
    def _update_height(self, dt):
        """Update height profile"""
        dhdt = self._calc_dhdt()
        self.h += dhdt * dt
        
        # Minimum height (film doesn't completely dry during simulation)
        h_min = 0.005 * (self.p.R**2 * self.p.theta0 / (2 * self.p.R))
        self.h = np.maximum(self.h, h_min)
        
        # Update contact angle
        self.theta = self.p.theta0 * (1 - self.t / self.t_f)
        self.theta = max(self.theta, 0.02)
        
    def _update_concentration(self, dt):
        """
        dc/dt + u_bar_r dc/dr = (D/rh) d_r(rh d_r c) + cJ/(rhoh) - S_dep/h
        
        Advection: upwind (since u > 0)
        Diffusion: central difference
        Evaporation: limited to prevent blow-up
        """
        p = self.p
        c_new = self.c.copy()
        
        # Minimum height for stability
        h_min = 0.01 * self.h[0]
        
        for i in range(1, p.Nr - 1):
            r_i = self.r[i]
            h_i = max(self.h[i], h_min)
            c_i = self.c[i]
            u_i = self.u[i]
            J_i = self.J[i]
            
            # === Advection (upwind, u > 0) ===
            if u_i >= 0:
                dc_dr = (c_i - self.c[i-1]) / self.dr
            else:
                dc_dr = (self.c[i+1] - c_i) / self.dr
            adv = -u_i * dc_dr
            
            # === Diffusion: (D/rh) d_r(rh d_r c) ===
            r_p = self.r[i] + self.dr/2
            r_m = self.r[i] - self.dr/2
            h_p = max(0.5 * (self.h[i] + self.h[i+1]), h_min)
            h_m = max(0.5 * (self.h[i] + self.h[i-1]), h_min)
            
            dc_dr_p = (self.c[i+1] - self.c[i]) / self.dr
            dc_dr_m = (self.c[i] - self.c[i-1]) / self.dr
            
            flux_p = r_p * h_p * dc_dr_p
            flux_m = r_m * h_m * dc_dr_m
            
            diff = p.D * (flux_p - flux_m) / (r_i * h_i * self.dr)
            
            # === Evaporation concentration (LIMITED) ===
            # As h->0, this term grows; limit to prevent instability
            evap_rate = J_i / (p.rho * h_i)
            evap_rate = min(evap_rate, 10.0 / dt)  # Limit growth rate
            evap = c_i * evap_rate
            
            # === Deposition sink ===
            dep_rate = p.k_dep * exp(-h_i / p.h_star)
            dep = c_i * dep_rate
            
            # === Update with stability limit ===
            dc = dt * (adv + diff + evap - dep)
            c_new[i] = max(c_i + dc, 0)
            
            # Limit maximum concentration
            c_new[i] = min(c_new[i], 100.0)
        
        # Boundary conditions
        c_new[0] = c_new[1]                      # Symmetry: dc/dr = 0
        c_new[-1] = max(c_new[-2], 0)            # Edge: extrapolate
        
        self.c = c_new
        
    def _update_deposition(self, dt):
        """dsigma/dt = k_dep * c * exp(-h/h*)"""
        p = self.p
        h_reg = np.maximum(self.h, 0.01 * self.h[0])
        S_dep = p.k_dep * self.c * exp(-h_reg / p.h_star)
        self.sigma += S_dep * dt
        
    def _total_mass(self):
        """Total particle mass (suspended + deposited)"""
        suspended = np.trapz(2 * pi * self.r * self.h * self.c, self.r)
        deposited = np.trapz(2 * pi * self.r * self.sigma, self.r)
        return suspended + deposited
    
    def _adaptive_dt(self):
        """CFL condition for stability"""
        p = self.p
        
        # Base timestep
        dt_base = 0.001 * self.t_f
        
        # Advection limit (only if velocity is significant)
        u_max = np.max(np.abs(self.u))
        if u_max > 1e-10:
            dt_adv = p.dt_safety * self.dr / u_max
        else:
            dt_adv = dt_base
        
        # Diffusion limit
        dt_diff = 0.25 * self.dr**2 / (p.D + 1e-20)
        
        # Height evolution limit (more aggressive near end)
        tau = max(self.t_f - self.t, 0.01 * self.t_f)
        dt_h = 0.05 * tau
        
        return min(dt_adv, dt_diff, dt_h, dt_base)
    
    def step(self):
        """Single time step"""
        # Compute fields
        self.J = self._calc_evap()
        self._calc_velocity()
        
        # Adaptive timestep
        dt = self._adaptive_dt()
        
        # Update state
        self._update_concentration(dt)
        self._update_deposition(dt)
        self._update_height(dt)
        
        self.t += dt
        return dt
    
    def run(self, verbose=True, save_interval=100):
        """Run simulation to completion"""
        
        if verbose:
            print("="*60)
            print("COFFEE RING SIMULATION")
            print("="*60)
            print(f"  R = {self.p.R}, theta_0 = {np.degrees(self.p.theta0):.1f}deg")
            print(f"  t_f = {self.t_f:.4f}")
            print(f"  Pe = {self.u.max() * self.p.R / self.p.D:.0f}")
            print("="*60)
        
        # Storage
        history = {
            't': [], 'theta': [], 'h': [], 'c': [], 
            'sigma': [], 'u': [], 'mass': []
        }
        
        n_step = 0
        while self.t < self.t_end:
            dt = self.step()
            n_step += 1
            
            # Save data
            if n_step % save_interval == 0:
                history['t'].append(self.t)
                history['theta'].append(self.theta)
                history['h'].append(self.h.copy())
                history['c'].append(self.c.copy())
                history['sigma'].append(self.sigma.copy())
                history['u'].append(self.u.copy())
                history['mass'].append(self._total_mass())
                
                if verbose and n_step % (save_interval * 10) == 0:
                    progress = self.t / self.t_end * 100
                    print(f"  t = {self.t:.4f} ({progress:.1f}%), "
                          f"theta = {np.degrees(self.theta):.1f}deg, "
                          f"max(sigma) = {self.sigma.max():.3f}")
        
        # Final save
        history['t'].append(self.t)
        history['theta'].append(self.theta)
        history['h'].append(self.h.copy())
        history['c'].append(self.c.copy())
        history['sigma'].append(self.sigma.copy())
        history['u'].append(self.u.copy())
        history['mass'].append(self._total_mass())
        
        # Convert to arrays
        for key in history:
            history[key] = np.array(history[key])
        
        if verbose:
            print("="*60)
            print(f"COMPLETED: {n_step} steps")
            print(f"  Final deposition peak: {self.sigma.max():.3f}")
            print(f"  Ring metric (edge/center): {self._ring_metric():.2f}")
            print(f"  Mass conservation: {100*self._total_mass()/self.mass0:.1f}%")
            print("="*60)
        
        self.history = history
        return history
    
    def _ring_metric(self):
        """Ratio of edge to center deposition"""
        n = len(self.sigma)
        center = np.mean(self.sigma[:n//4]) + 1e-10
        edge = np.mean(self.sigma[-n//4:])
        return edge / center


# ==============================================================================
# VISUALIZATION
# ==============================================================================

class Visualizer:
    """Plotting utilities for coffee ring simulation"""
    
    def __init__(self, sim):
        self.sim = sim
        self.r = sim.r
        self.R = sim.p.R
        
    def plot_final_deposition(self, ax=None):
        """Final deposition profile"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(self.r/self.R, self.sim.sigma, 'b-', lw=2.5)
        ax.fill_between(self.r/self.R, 0, self.sim.sigma, alpha=0.3)
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Deposition $\sigma$', fontsize=14)
        ax.set_title('Coffee Ring Pattern', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_height_evolution(self, ax=None):
        """Height profile evolution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        hist = self.sim.history
        n_curves = min(6, len(hist['t']))
        indices = np.linspace(0, len(hist['t'])-1, n_curves, dtype=int)
        
        for i in indices:
            t = hist['t'][i]
            h = hist['h'][i]
            ax.plot(self.r/self.R, h, lw=2, alpha=0.8,
                   label=f"t = {t:.3f}")
        
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Height $h$', fontsize=14)
        ax.set_title('Droplet Height Evolution', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_concentration_evolution(self, ax=None):
        """Concentration profile evolution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        hist = self.sim.history
        n_curves = min(6, len(hist['t']))
        indices = np.linspace(0, len(hist['t'])-1, n_curves, dtype=int)
        
        for i in indices:
            t = hist['t'][i]
            c = hist['c'][i]
            ax.plot(self.r/self.R, c, lw=2, alpha=0.8,
                   label=f"t = {t:.3f}")
        
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Concentration $c$', fontsize=14)
        ax.set_title('Particle Concentration Evolution', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_deposition_evolution(self, ax=None):
        """Deposition profile evolution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        hist = self.sim.history
        n_curves = min(6, len(hist['t']))
        indices = np.linspace(0, len(hist['t'])-1, n_curves, dtype=int)
        
        for i in indices:
            t = hist['t'][i]
            sigma = hist['sigma'][i]
            ax.plot(self.r/self.R, sigma, lw=2, alpha=0.8,
                   label=f"t = {t:.3f}")
        
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Deposition $\sigma$', fontsize=14)
        ax.set_title('Deposition Evolution', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_velocity_evolution(self, ax=None):
        """Velocity profile evolution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        hist = self.sim.history
        n_curves = min(6, len(hist['t']))
        indices = np.linspace(0, len(hist['t'])-1, n_curves, dtype=int)
        
        for i in indices:
            t = hist['t'][i]
            u = hist['u'][i]
            ax.plot(self.r/self.R, u, lw=2, alpha=0.8,
                   label=f"t = {t:.3f}")
        
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Velocity $\bar{u}_r$', fontsize=14)
        ax.set_title('Radial Velocity Evolution', fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_2d_pattern(self, ax=None):
        """2D visualization of deposition pattern"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create 2D grid
        n_grid = 200
        x = np.linspace(-self.R, self.R, n_grid)
        y = np.linspace(-self.R, self.R, n_grid)
        X, Y = np.meshgrid(x, y)
        R_grid = np.sqrt(X**2 + Y**2)
        
        # Interpolate deposition to 2D
        Z = np.interp(R_grid.flatten(), self.r, self.sim.sigma)
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > self.R] = 0
        
        im = ax.imshow(Z, extent=[-self.R, self.R, -self.R, self.R],
                       origin='lower', cmap='YlOrBr')
        
        circle = plt.Circle((0, 0), self.R, fill=False, color='k', lw=2)
        ax.add_patch(circle)
        
        ax.set_xlabel(r'$x/R$', fontsize=14)
        ax.set_ylabel(r'$y/R$', fontsize=14)
        ax.set_title('Coffee Ring (Top View)', fontsize=16, fontweight='bold')
        ax.set_aspect('equal')
        
        cbar = plt.colorbar(im, ax=ax, label=r'$\sigma$', shrink=0.8)
        return ax
    
    def plot_initial_distribution(self, ax=None):
        """Plot initial particle concentration distribution"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        
        # Get initial concentration from history
        if self.sim.history and 'c' in self.sim.history and len(self.sim.history['c']) > 0:
            c_init = self.sim.history['c'][0]
        else:
            c_init = self.sim.c  # Current (might not be initial if sim ran)
        
        ax.fill_between(self.r_norm, 0, c_init, alpha=0.4, color='purple')
        ax.plot(self.r_norm, c_init, 'purple', lw=2.5, label='Initial c(r)')
        ax.set_xlabel(r'$r/R$', fontsize=14)
        ax.set_ylabel(r'Initial Concentration $c_0$', fontsize=14)
        ax.set_title(f'Initial Particle Distribution ({self.sim.p.init_type})', 
                     fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(c_init) * 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig
    
    def plot_summary(self):
        """4-panel summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        self.plot_height_evolution(axes[0, 0])
        self.plot_concentration_evolution(axes[0, 1])
        self.plot_deposition_evolution(axes[1, 0])
        self.plot_final_deposition(axes[1, 1])
        
        plt.tight_layout()
        return fig
    
    def plot_full_summary(self):
        """6-panel complete summary"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 9))
        
        self.plot_height_evolution(axes[0, 0])
        self.plot_velocity_evolution(axes[0, 1])
        self.plot_concentration_evolution(axes[0, 2])
        self.plot_deposition_evolution(axes[1, 0])
        self.plot_final_deposition(axes[1, 1])
        self.plot_2d_pattern(axes[1, 2])
        
        plt.tight_layout()
        return fig


# ==============================================================================
# ANIMATIONS
# ==============================================================================

def animate_height(sim):
    """Animate height evolution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    hist = sim.history
    
    line, = ax.plot([], [], 'b-', lw=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(hist['h']) * 1.1)
    ax.set_xlabel(r'$r/R$', fontsize=14)
    ax.set_ylabel(r'Height $h$', fontsize=14)
    ax.set_title('Droplet Height', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(i):
        line.set_data(sim.r / sim.p.R, hist['h'][i])
        time_text.set_text(f"t = {hist['t'][i]:.4f}")
        return line, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(hist['t']),
                         interval=100, blit=True)
    plt.tight_layout()
    return fig, anim


def animate_concentration(sim):
    """Animate concentration evolution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    hist = sim.history
    
    line, = ax.plot([], [], 'r-', lw=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(hist['c']) * 1.1)
    ax.set_xlabel(r'$r/R$', fontsize=14)
    ax.set_ylabel(r'Concentration $c$', fontsize=14)
    ax.set_title('Particle Concentration', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(i):
        line.set_data(sim.r / sim.p.R, hist['c'][i])
        time_text.set_text(f"t = {hist['t'][i]:.4f}")
        return line, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(hist['t']),
                         interval=100, blit=True)
    plt.tight_layout()
    return fig, anim


def animate_deposition(sim):
    """Animate deposition evolution"""
    fig, ax = plt.subplots(figsize=(8, 5))
    hist = sim.history
    
    line, = ax.plot([], [], 'g-', lw=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(hist['sigma'][-1]) * 1.1)
    ax.set_xlabel(r'$r/R$', fontsize=14)
    ax.set_ylabel(r'Deposition $\sigma$', fontsize=14)
    ax.set_title('Particle Deposition', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(i):
        line.set_data(sim.r / sim.p.R, hist['sigma'][i])
        time_text.set_text(f"t = {hist['t'][i]:.4f}")
        return line, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(hist['t']),
                         interval=100, blit=True)
    plt.tight_layout()
    return fig, anim


def animate_2d(sim):
    """Animate 2D deposition pattern"""
    fig, ax = plt.subplots(figsize=(8, 8))
    hist = sim.history
    R = sim.p.R
    
    # Grid
    n_grid = 150
    x = np.linspace(-R, R, n_grid)
    y = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    
    # Initial image
    Z_init = np.zeros_like(R_grid)
    im = ax.imshow(Z_init, extent=[-R, R, -R, R], origin='lower',
                   cmap='YlOrBr', vmin=0, vmax=np.max(hist['sigma'][-1]))
    
    circle = plt.Circle((0, 0), R, fill=False, color='white', lw=2)
    ax.add_patch(circle)
    
    ax.set_xlabel(r'$x/R$', fontsize=14)
    ax.set_ylabel(r'$y/R$', fontsize=14)
    ax.set_title('Coffee Ring Formation', fontsize=16, fontweight='bold')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, label=r'$\sigma$', shrink=0.8)
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        color='white', fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
    
    def animate(i):
        Z = np.interp(R_grid.flatten(), sim.r, hist['sigma'][i])
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > R] = 0
        im.set_data(Z)
        time_text.set_text(f"t = {hist['t'][i]:.4f}")
        return [im, time_text]
    
    anim = FuncAnimation(fig, animate, frames=len(hist['t']),
                         interval=150, blit=False)
    plt.tight_layout()
    return fig, anim


def animate_full(sim, save_path=None, fps=15):
    """
    Comprehensive animation showing coffee ring formation.
    
    4-panel animation:
    - Top left: Height profile h(r)
    - Top right: Concentration c(r)
    - Bottom left: Deposition sigma(r)
    - Bottom right: 2D top view of ring formation
    
    Parameters:
    -----------
    sim : CoffeeRingSim
        Completed simulation with history
    save_path : str, optional
        Path to save animation (e.g., 'coffee_ring.mp4' or 'coffee_ring.gif')
    fps : int
        Frames per second for saved animation
    
    Returns:
    --------
    fig, anim : Figure and FuncAnimation objects
    """
    hist = sim.history
    R = sim.p.R
    n_frames = len(hist['t'])
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])  # Height
    ax2 = fig.add_subplot(gs[0, 1])  # Concentration
    ax3 = fig.add_subplot(gs[1, 0])  # Deposition profile
    ax4 = fig.add_subplot(gs[1, 1])  # 2D view
    
    r_norm = sim.r / R
    
    # === Panel 1: Height ===
    h_max = np.max(hist['h'][0]) * 1.1
    line_h, = ax1.plot([], [], 'b-', lw=2.5)
    fill_h = ax1.fill_between([], [], alpha=0.3, color='blue')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, h_max)
    ax1.set_xlabel(r'$r/R$', fontsize=12)
    ax1.set_ylabel(r'Height $h$', fontsize=12)
    ax1.set_title('Droplet Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: Concentration ===
    c_max = min(np.max([np.max(c) for c in hist['c']]), 100) * 1.1
    line_c, = ax2.plot([], [], 'r-', lw=2.5)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, c_max)
    ax2.set_xlabel(r'$r/R$', fontsize=12)
    ax2.set_ylabel(r'Concentration $c$', fontsize=12)
    ax2.set_title('Particle Concentration', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Deposition profile ===
    sigma_max = np.max(hist['sigma'][-1]) * 1.1
    line_s, = ax3.plot([], [], 'g-', lw=2.5)
    fill_s = ax3.fill_between([], [], alpha=0.3, color='green')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, sigma_max)
    ax3.set_xlabel(r'$r/R$', fontsize=12)
    ax3.set_ylabel(r'Deposition $\sigma$', fontsize=12)
    ax3.set_title('Deposited Particles', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: 2D view ===
    n_grid = 150
    x = np.linspace(-R, R, n_grid)
    y = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    
    Z_init = np.zeros_like(R_grid)
    im = ax4.imshow(Z_init, extent=[-1, 1, -1, 1], origin='lower',
                    cmap='YlOrBr', vmin=0, vmax=np.max(hist['sigma'][-1]))
    circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
    ax4.add_patch(circle)
    ax4.set_xlabel(r'$x/R$', fontsize=12)
    ax4.set_ylabel(r'$y/R$', fontsize=12)
    ax4.set_title('Coffee Ring (Top View)', fontsize=14, fontweight='bold')
    ax4.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label(r'$\sigma$', fontsize=12)
    
    # Time display
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=14,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Progress bar axes
    ax_prog = fig.add_axes([0.25, 0.96, 0.5, 0.02])
    ax_prog.set_xlim(0, 1)
    ax_prog.set_ylim(0, 1)
    ax_prog.axis('off')
    prog_bar = ax_prog.barh(0.5, 0, height=0.8, color='steelblue', alpha=0.7)
    ax_prog.text(-0.02, 0.5, '0%', ha='right', va='center', fontsize=10)
    ax_prog.text(1.02, 0.5, '100%', ha='left', va='center', fontsize=10)
    
    def init():
        line_h.set_data([], [])
        line_c.set_data([], [])
        line_s.set_data([], [])
        return [line_h, line_c, line_s, im]
    
    def animate(i):
        # Update height
        line_h.set_data(r_norm, hist['h'][i])
        
        # Update height fill (recreate)
        ax1.collections.clear()
        ax1.fill_between(r_norm, 0, hist['h'][i], alpha=0.3, color='blue')
        
        # Update concentration
        line_c.set_data(r_norm, hist['c'][i])
        
        # Update deposition profile
        line_s.set_data(r_norm, hist['sigma'][i])
        ax3.collections.clear()
        ax3.fill_between(r_norm, 0, hist['sigma'][i], alpha=0.3, color='green')
        
        # Update 2D view
        Z = np.interp(R_grid.flatten(), sim.r, hist['sigma'][i])
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > R] = np.nan
        im.set_data(Z)
        
        # Update time and progress
        t = hist['t'][i]
        progress = t / sim.t_f
        time_text.set_text(f't = {t:.4f} / {sim.t_f:.4f}  ({100*progress:.1f}% dried)')
        
        # Update progress bar
        prog_bar[0].set_width(progress)
        
        return [line_h, line_c, line_s, im, time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                         interval=1000//fps, blit=False)
    
    # Save if requested
    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=fps)
        else:
            anim.save(save_path, writer='ffmpeg', fps=fps)
        print("  Done!")
    
    return fig, anim


def create_ring_animation_gif(sim, filename='coffee_ring_animation.gif', fps=10):
    """
    Create a simple GIF animation of coffee ring formation.
    
    This version uses a simpler approach that works without ffmpeg.
    """
    import matplotlib
    matplotlib.use('Agg')
    
    hist = sim.history
    R = sim.p.R
    n_frames = len(hist['t'])
    
    # Grid for 2D view
    n_grid = 100
    x = np.linspace(-R, R, n_grid)
    y = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    
    # Create frames
    frames = []
    
    print(f"  Generating {n_frames} frames...")
    
    for i in range(n_frames):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
        
        # Left: Deposition profile
        ax1 = axes[0]
        ax1.fill_between(sim.r/R, 0, hist['sigma'][i], alpha=0.6, color='saddlebrown')
        ax1.plot(sim.r/R, hist['sigma'][i], 'k-', lw=2)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, np.max(hist['sigma'][-1]) * 1.1)
        ax1.set_xlabel(r'$r/R$', fontsize=12)
        ax1.set_ylabel(r'Deposition $\sigma$', fontsize=12)
        ax1.set_title(f'Radial Profile (t = {hist["t"][i]:.3f})', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Right: 2D view
        ax2 = axes[1]
        Z = np.interp(R_grid.flatten(), sim.r, hist['sigma'][i])
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > R] = np.nan
        
        im = ax2.imshow(Z, extent=[-1, 1, -1, 1], origin='lower',
                        cmap='YlOrBr', vmin=0, vmax=np.max(hist['sigma'][-1]))
        circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
        ax2.add_patch(circle)
        ax2.set_xlabel(r'$x/R$', fontsize=12)
        ax2.set_ylabel(r'$y/R$', fontsize=12)
        ax2.set_title('Top View', fontsize=12)
        ax2.set_aspect('equal')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # Progress indicator
        progress = hist['t'][i] / sim.t_f
        fig.suptitle(f'Coffee Ring Formation - {100*progress:.0f}% Complete', 
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save frame to buffer (compatible with newer matplotlib)
        fig.canvas.draw()
        # Try different methods for compatibility
        try:
            # Newer matplotlib
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3]  # Drop alpha channel
        except AttributeError:
            try:
                # Older matplotlib
                frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            except AttributeError:
                # Fallback: save to file and read back
                import io
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                from PIL import Image
                frame = np.array(Image.open(buf))[:, :, :3]
        
        frames.append(frame.copy())
        plt.close(fig)
        
        if (i + 1) % 10 == 0:
            print(f"    Frame {i+1}/{n_frames}")
    
    # Save as GIF using PIL
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(filename, save_all=True, append_images=imgs[1:], 
                     duration=1000//fps, loop=0)
        print(f"  Saved animation: {filename}")
    except ImportError:
        print("  PIL not available. Saving frames as PNG instead...")
        for i, frame in enumerate(frames):
            plt.imsave(f'frame_{i:03d}.png', frame)
        print(f"  Saved {len(frames)} frames as frame_XXX.png")


# ==============================================================================
# PRESET CONFIGURATIONS
# ==============================================================================

def preset_standard():
    """Standard coffee ring parameters"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='uniform'
    )

def preset_high_Pe():
    """High Peclet number (strong advection)"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.0001,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='uniform'
    )

def preset_low_Pe():
    """Low Peclet number (significant diffusion)"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.01,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='uniform'
    )

def preset_thin_drop():
    """Thin droplet (small contact angle)"""
    return Params(
        R=1.0, theta0=0.2, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.01,
        Nr=200, t_end_frac=0.95,
        init_type='uniform'
    )

def preset_thick_drop():
    """Thick droplet (large contact angle)"""
    return Params(
        R=1.0, theta0=0.8, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.05,
        Nr=200, t_end_frac=0.95,
        init_type='uniform'
    )

def preset_quick():
    """Quick test run"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.001,
        J0=0.2, k_dep=0.2, h_star=0.02,
        Nr=100, t_end_frac=0.90,
        init_type='uniform'
    )

def preset_random_particles():
    """Random initial particle distribution"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='random',
        init_seed=42
    )

def preset_center_particles():
    """Particles concentrated at center"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='center',
        init_width=0.2
    )

def preset_blob_particles():
    """Random blobs of particles"""
    return Params(
        R=1.0, theta0=0.5, rho=1.0, D=0.001,
        J0=0.1, k_dep=0.1, h_star=0.02,
        Nr=200, t_end_frac=0.95,
        init_type='blobs',
        init_seed=123
    )


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import sys
    
    # Parse command line for demo type
    demo = sys.argv[1] if len(sys.argv) > 1 else 'standard'
    
    print("="*60)
    print("COFFEE RING SIMULATION")
    print("="*60)
    
    # Select preset based on argument
    if demo == 'random':
        params = preset_random_particles()
        print("  Mode: Random initial particles")
    elif demo == 'center':
        params = preset_center_particles()
        print("  Mode: Center-concentrated particles")
    elif demo == 'blobs':
        params = preset_blob_particles()
        print("  Mode: Random blob distribution")
    elif demo == 'quick':
        params = preset_quick()
        print("  Mode: Quick test")
    else:
        params = preset_standard()
        print("  Mode: Standard uniform distribution")
    
    print("="*60)
    
    # Run simulation
    print("\nRunning simulation...")
    sim = CoffeeRingSim(params)
    history = sim.run(verbose=True, save_interval=20)  # More frames for animation
    
    # Create plots
    print("\nGenerating plots...")
    viz = Visualizer(sim)
    
    fig1 = viz.plot_summary()
    fig1.savefig('coffee_ring_summary.png', dpi=150, bbox_inches='tight')
    print("  Saved: coffee_ring_summary.png")
    
    fig2 = viz.plot_full_summary()
    fig2.savefig('coffee_ring_full.png', dpi=150, bbox_inches='tight')
    print("  Saved: coffee_ring_full.png")
    
    # Create animation (GIF)
    print("\nGenerating animation...")
    try:
        create_ring_animation_gif(sim, 'coffee_ring_animation.gif', fps=10)
    except Exception as e:
        print(f"  Animation failed: {e}")
    
    # Print statistics
    print("\n" + "="*60)
    print("COFFEE RING STATISTICS")
    print("="*60)
    n = len(sim.sigma)
    center_dep = np.mean(sim.sigma[:n//4])
    edge_dep = np.mean(sim.sigma[-n//4:])
    print(f"  Initial distribution:         {params.init_type}")
    print(f"  Center deposition (r < R/4):  {center_dep:.4f}")
    print(f"  Edge deposition (r > 3R/4):   {edge_dep:.4f}")
    print(f"  Ring enhancement factor:      {edge_dep/(center_dep+1e-10):.2f}")
    print(f"  Peak location (r/R):          {sim.r[np.argmax(sim.sigma)]/sim.p.R:.3f}")
    print("="*60)
    
    print("\nUsage examples:")
    print("  python coffee_ring_depth_averaged.py           # Standard")
    print("  python coffee_ring_depth_averaged.py random    # Random particles")
    print("  python coffee_ring_depth_averaged.py center    # Center concentrated")
    print("  python coffee_ring_depth_averaged.py blobs     # Random blobs")
