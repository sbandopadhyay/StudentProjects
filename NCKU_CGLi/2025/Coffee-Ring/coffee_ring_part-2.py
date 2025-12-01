"""
Two-Layer Coffee Ring Simulation with Marangoni Effects
========================================================

Mathematical Model:
- Two-layer (surface/bulk) to capture vertical recirculation
- Marangoni flow from surface tension gradients
- Capillary flow from evaporation-driven replenishment

References:
- Deegan et al. (1997) Nature - Coffee ring discovery
- Hu & Larson (2002, 2005, 2006) - Evaporation and Marangoni flow
- Popov (2005) Phys. Rev. E - Analytical theory
"""

import numpy as np
from numpy import pi, sqrt, exp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass


@dataclass
class Params:
    R: float = 1.0
    theta0: float = 0.5
    rho: float = 1.0
    D: float = 0.001
    J0: float = 0.1
    k_dep: float = 0.1
    h_star: float = 0.02
    Ma: float = 0.0
    Nr: int = 200
    eps: float = 0.02
    dt_safety: float = 0.5
    t_end_frac: float = 0.95
    init_type: str = 'uniform'
    init_center: float = 0.0
    init_width: float = 0.2
    init_seed: int = None


class CoffeeRingMarangoni:
    
    def __init__(self, params=None):
        self.p = params or Params()
        self._setup()
    
    def _setup(self):
        p = self.p
        self.r = np.linspace(p.eps * p.R, p.R, p.Nr)
        self.dr = self.r[1] - self.r[0]
        self.r_norm = self.r / p.R
        self.theta = p.theta0
        self.h = self._calc_height()
        self.h0 = p.R * p.theta0 / 2
        V0 = np.trapz(2 * pi * self.r * self.h, self.r)
        self.J = self._calc_evap()
        dVdt = np.trapz(2 * pi * self.r * self.J / p.rho, self.r)
        self.t_f = V0 / dVdt
        self.U_cap = p.R / self.t_f
        self.t = 0.0
        self.t_end = p.t_end_frac * self.t_f
        self.u_cap = np.zeros(p.Nr)
        self.u_s_ma = np.zeros(p.Nr)
        self.u_s = np.zeros(p.Nr)
        self.u_b = np.zeros(p.Nr)
        self.c_s, self.c_b = self._init_concentration()
        self.sigma = np.zeros(p.Nr)
        self.history = {'t': [], 'theta': [], 'h': [], 'c_s': [], 'c_b': [], 
                        'sigma': [], 'u_s': [], 'u_b': [], 'u_cap': [], 'u_s_ma': []}
    
    def _init_concentration(self):
        p = self.p
        r_norm = self.r / p.R
        if p.init_seed is not None:
            np.random.seed(p.init_seed)
        if p.init_type == 'uniform':
            c = np.ones(p.Nr)
        elif p.init_type == 'gaussian':
            c = np.exp(-((r_norm - p.init_center)**2) / (2 * p.init_width**2))
            c = np.maximum(c, 0.01)
        elif p.init_type == 'center':
            c = np.exp(-(r_norm**2) / (2 * p.init_width**2))
            c = np.maximum(c, 0.01)
        elif p.init_type == 'random':
            noise = np.random.randn(p.Nr) * 0.3
            kernel = np.exp(-np.arange(-10, 11)**2 / 20)
            kernel /= kernel.sum()
            noise = np.convolve(noise, kernel, mode='same')
            c = 1.0 + noise
            c = np.maximum(c, 0.1)
        elif p.init_type == 'blobs':
            n_blobs = np.random.randint(3, 7)
            c = 0.1 * np.ones(p.Nr)
            for _ in range(n_blobs):
                r_blob = np.random.uniform(0.1, 0.8)
                w_blob = np.random.uniform(0.05, 0.15)
                amp = np.random.uniform(0.5, 2.0)
                c += amp * np.exp(-((r_norm - r_blob)**2) / (2 * w_blob**2))
        else:
            c = np.ones(p.Nr)
        return c.copy(), c.copy()
    
    def _calc_height(self):
        p = self.p
        h = (p.R**2 - self.r**2) * self.theta / (2 * p.R)
        return np.maximum(h, 1e-10)
    
    def _calc_evap(self):
        p = self.p
        lam = (pi - 2*self.theta) / (2*pi - 2*self.theta)
        arg = 1 - self.r_norm**2
        arg = np.maximum(arg, p.eps)
        J = p.J0 * arg**(-lam)
        return J
    
    def _calc_capillary_velocity(self):
        p = self.p
        tau = max(self.t_f - self.t, 0.01 * self.t_f)
        integrand = (self.h / tau - self.J / p.rho) * self.r
        integral = np.zeros(p.Nr)
        for i in range(1, p.Nr):
            integral[i] = integral[i-1] + 0.5 * (integrand[i] + integrand[i-1]) * self.dr
        h_reg = np.maximum(self.h, 0.01 * self.h0)
        self.u_cap = integral / (self.r * h_reg)
        self.u_cap[0] = 0
        u_scale = p.R / self.t_f
        self.u_cap = np.clip(self.u_cap, -u_scale, 50 * u_scale)
        self.u_cap[-5:] = np.linspace(self.u_cap[-6], self.u_cap[-6] * 1.5, 5)
    
    def _calc_marangoni_velocity(self):
        p = self.p
        if abs(p.Ma) < 1e-10:
            self.u_s_ma = np.zeros(p.Nr)
            return
        g_r = (1 - self.r_norm**2 + p.eps)**(-0.3)
        h_ratio = self.h / self.h0
        self.u_s_ma = -p.Ma * self.U_cap * h_ratio * g_r
        u_scale = abs(p.Ma) * self.U_cap * 2
        self.u_s_ma = np.clip(self.u_s_ma, -u_scale, u_scale)
    
    def _calc_layer_velocities(self):
        self.u_s = (11.0/8.0) * self.u_cap + (3.0/4.0) * self.u_s_ma
        self.u_b = (5.0/8.0) * self.u_cap + (1.0/4.0) * self.u_s_ma
    
    def _calc_exchange_rate(self):
        p = self.p
        h_reg = np.maximum(self.h, 0.01 * self.h0)
        k_diff = 16 * p.D / h_reg**2
        k_adv = 2 * np.abs(self.u_s_ma) / p.R
        k_ex = k_diff + k_adv
        k_max = 100.0 / max(self._adaptive_dt_base(), 1e-6)
        return np.minimum(k_ex, k_max)
    
    def _adaptive_dt_base(self):
        p = self.p
        dt_base = 0.001 * self.t_f
        u_max = max(np.max(np.abs(self.u_s)), np.max(np.abs(self.u_b)), 1e-10)
        dt_adv = p.dt_safety * self.dr / u_max
        dt_diff = 0.25 * self.dr**2 / (p.D + 1e-20)
        tau = max(self.t_f - self.t, 0.01 * self.t_f)
        dt_h = 0.05 * tau
        return min(dt_adv, dt_diff, dt_h, dt_base)
    
    def _update_concentration(self, dt):
        p = self.p
        k_ex = self._calc_exchange_rate()
        c_s_new = self.c_s.copy()
        c_b_new = self.c_b.copy()
        h_min = 0.01 * self.h0
        for i in range(1, p.Nr - 1):
            r_i = self.r[i]
            h_i = max(self.h[i], h_min)
            c_s_i = self.c_s[i]
            c_b_i = self.c_b[i]
            u_s_i = self.u_s[i]
            u_b_i = self.u_b[i]
            J_i = self.J[i]
            k_i = k_ex[i]
            if u_s_i >= 0:
                dc_s_dr = (c_s_i - self.c_s[i-1]) / self.dr
            else:
                dc_s_dr = (self.c_s[i+1] - c_s_i) / self.dr
            adv_s = -u_s_i * dc_s_dr
            if u_b_i >= 0:
                dc_b_dr = (c_b_i - self.c_b[i-1]) / self.dr
            else:
                dc_b_dr = (self.c_b[i+1] - c_b_i) / self.dr
            adv_b = -u_b_i * dc_b_dr
            r_p = self.r[i] + self.dr/2
            r_m = self.r[i] - self.dr/2
            h_p = max(0.5 * (self.h[i] + self.h[i+1]), h_min)
            h_m = max(0.5 * (self.h[i] + self.h[i-1]), h_min)
            dc_s_dr_p = (self.c_s[i+1] - self.c_s[i]) / self.dr
            dc_s_dr_m = (self.c_s[i] - self.c_s[i-1]) / self.dr
            flux_s_p = r_p * h_p * dc_s_dr_p
            flux_s_m = r_m * h_m * dc_s_dr_m
            diff_s = p.D * (flux_s_p - flux_s_m) / (r_i * h_i * self.dr)
            dc_b_dr_p = (self.c_b[i+1] - self.c_b[i]) / self.dr
            dc_b_dr_m = (self.c_b[i] - self.c_b[i-1]) / self.dr
            flux_b_p = r_p * h_p * dc_b_dr_p
            flux_b_m = r_m * h_m * dc_b_dr_m
            diff_b = p.D * (flux_b_p - flux_b_m) / (r_i * h_i * self.dr)
            evap_rate = J_i / (p.rho * h_i)
            evap_rate = min(evap_rate, 10.0 / dt)
            evap_s = c_s_i * evap_rate
            exchange_s = k_i * (c_b_i - c_s_i)
            exchange_b = k_i * (c_s_i - c_b_i)
            dep_rate = p.k_dep * exp(-h_i / p.h_star)
            dep_b = 2.0 * c_b_i * dep_rate
            dc_s = dt * (adv_s + diff_s + evap_s + exchange_s)
            dc_b = dt * (adv_b + diff_b + exchange_b - dep_b)
            c_s_new[i] = max(c_s_i + dc_s, 0)
            c_b_new[i] = max(c_b_i + dc_b, 0)
            c_s_new[i] = min(c_s_new[i], 100.0)
            c_b_new[i] = min(c_b_new[i], 100.0)
        c_s_new[0] = c_s_new[1]
        c_b_new[0] = c_b_new[1]
        c_s_new[-1] = max(c_s_new[-2], 0)
        c_b_new[-1] = max(c_b_new[-2], 0)
        self.c_s = c_s_new
        self.c_b = c_b_new
    
    def _update_deposition(self, dt):
        p = self.p
        h_reg = np.maximum(self.h, 0.01 * self.h0)
        S_dep = p.k_dep * self.c_b * exp(-h_reg / p.h_star)
        self.sigma += S_dep * dt
    
    def _update_height(self, dt):
        tau = max(self.t_f - self.t, 0.01 * self.t_f)
        self.h = self.h * (1 - dt / tau)
        h_min = 0.005 * self.h0
        self.h = np.maximum(self.h, h_min)
        self.theta = self.p.theta0 * (1 - self.t / self.t_f)
        self.theta = max(self.theta, 0.02)
    
    def step(self):
        self.J = self._calc_evap()
        self._calc_capillary_velocity()
        self._calc_marangoni_velocity()
        self._calc_layer_velocities()
        dt = self._adaptive_dt_base()
        self._update_concentration(dt)
        self._update_deposition(dt)
        self._update_height(dt)
        self.t += dt
        return dt
    
    def run(self, verbose=True, save_interval=50):
        if verbose:
            print("="*60)
            print("COFFEE RING SIMULATION WITH MARANGONI EFFECT")
            print("="*60)
            print(f"  R = {self.p.R}, theta_0 = {np.degrees(self.p.theta0):.1f}deg")
            print(f"  Ma = {self.p.Ma}")
            print(f"  t_f = {self.t_f:.4f}")
            print("="*60)
        n_step = 0
        while self.t < self.t_end:
            dt = self.step()
            n_step += 1
            if n_step % save_interval == 0:
                self.history['t'].append(self.t)
                self.history['theta'].append(self.theta)
                self.history['h'].append(self.h.copy())
                self.history['c_s'].append(self.c_s.copy())
                self.history['c_b'].append(self.c_b.copy())
                self.history['sigma'].append(self.sigma.copy())
                self.history['u_s'].append(self.u_s.copy())
                self.history['u_b'].append(self.u_b.copy())
                self.history['u_cap'].append(self.u_cap.copy())
                self.history['u_s_ma'].append(self.u_s_ma.copy())
            if verbose and n_step % (save_interval * 5) == 0:
                pct = 100 * self.t / self.t_f
                print(f"  t = {self.t:.4f} ({pct:.1f}%), theta = {np.degrees(self.theta):.1f}deg, max(sigma) = {self.sigma.max():.3f}")
        self.history['t'].append(self.t)
        self.history['theta'].append(self.theta)
        self.history['h'].append(self.h.copy())
        self.history['c_s'].append(self.c_s.copy())
        self.history['c_b'].append(self.c_b.copy())
        self.history['sigma'].append(self.sigma.copy())
        self.history['u_s'].append(self.u_s.copy())
        self.history['u_b'].append(self.u_b.copy())
        self.history['u_cap'].append(self.u_cap.copy())
        self.history['u_s_ma'].append(self.u_s_ma.copy())
        if verbose:
            print("="*60)
            print(f"COMPLETED: {n_step} steps")
            n = len(self.sigma)
            center_dep = np.mean(self.sigma[:n//4])
            edge_dep = np.mean(self.sigma[-n//4:])
            print(f"  Final deposition peak: {self.sigma.max():.3f}")
            print(f"  Ring metric (edge/center): {edge_dep/(center_dep+1e-10):.2f}")
            print("="*60)
        return self.history


class Visualizer:
    
    def __init__(self, sim):
        self.sim = sim
        self.r_norm = sim.r / sim.p.R
    
    def plot_velocity_comparison(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        hist = self.sim.history
        n_times = len(hist['t'])
        indices = [0, n_times//3, 2*n_times//3, -1]
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(indices)))
        for idx, c in zip(indices, colors):
            t = hist['t'][idx]
            u_s = hist['u_s'][idx]
            u_b = hist['u_b'][idx]
            ax.plot(self.r_norm, u_s, '-', color=c, lw=2, label=f't={t:.3f} (surface)')
            ax.plot(self.r_norm, u_b, '--', color=c, lw=1.5)
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(r'$r/R$', fontsize=12)
        ax.set_ylabel(r'Velocity', fontsize=12)
        ax.set_title('Layer Velocities (solid=surface, dashed=bulk)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_concentration_layers(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        hist = self.sim.history
        n_times = len(hist['t'])
        indices = [0, n_times//3, 2*n_times//3, -1]
        colors = plt.cm.plasma(np.linspace(0, 0.9, len(indices)))
        for idx, c in zip(indices, colors):
            t = hist['t'][idx]
            c_s = hist['c_s'][idx]
            c_b = hist['c_b'][idx]
            ax.plot(self.r_norm, c_s, '-', color=c, lw=2, label=f't={t:.3f}')
            ax.plot(self.r_norm, c_b, '--', color=c, lw=1.5)
        ax.set_xlabel(r'$r/R$', fontsize=12)
        ax.set_ylabel(r'Concentration $c$', fontsize=12)
        ax.set_title('Concentration (solid=surface, dashed=bulk)', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_deposition_evolution(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        hist = self.sim.history
        n_times = len(hist['t'])
        indices = np.linspace(0, n_times-1, 6, dtype=int)
        colors = plt.cm.Oranges(np.linspace(0.3, 1.0, len(indices)))
        for idx, c in zip(indices, colors):
            t = hist['t'][idx]
            sigma = hist['sigma'][idx]
            ax.plot(self.r_norm, sigma, color=c, lw=2, label=f't = {t:.3f}')
        ax.set_xlabel(r'$r/R$', fontsize=12)
        ax.set_ylabel(r'Deposition $\sigma$', fontsize=12)
        ax.set_title('Deposition Evolution', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_final_deposition(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        ax.fill_between(self.r_norm, 0, self.sim.sigma, alpha=0.4, color='saddlebrown')
        ax.plot(self.r_norm, self.sim.sigma, 'k-', lw=2)
        ax.set_xlabel(r'$r/R$', fontsize=12)
        ax.set_ylabel(r'Deposition $\sigma$', fontsize=12)
        ax.set_title(f'Final Pattern (Ma = {self.sim.p.Ma})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_2d_pattern(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        R = self.sim.p.R
        n_grid = 150
        x = np.linspace(-R, R, n_grid)
        y = np.linspace(-R, R, n_grid)
        X, Y = np.meshgrid(x, y)
        R_grid = np.sqrt(X**2 + Y**2)
        sigma_interp = np.interp(R_grid.flatten(), self.sim.r, self.sim.sigma)
        sigma_interp = sigma_interp.reshape(R_grid.shape)
        sigma_interp[R_grid > R] = np.nan
        im = ax.imshow(sigma_interp, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrBr')
        circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
        ax.add_patch(circle)
        ax.set_xlabel(r'$x/R$', fontsize=12)
        ax.set_ylabel(r'$y/R$', fontsize=12)
        ax.set_title(f'Top View (Ma = {self.sim.p.Ma})', fontsize=12, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$\sigma$')
        return ax
    
    def plot_marangoni_effect(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        hist = self.sim.history
        if len(hist['u_cap']) > 0:
            idx = len(hist['t']) // 2
            u_cap = hist['u_cap'][idx]
            u_s_ma = hist['u_s_ma'][idx]
            u_s = hist['u_s'][idx]
            u_b = hist['u_b'][idx]
            ax.plot(self.r_norm, u_cap, 'b-', lw=2, label=r'$\bar{u}_r^{cap}$ (capillary)')
            ax.plot(self.r_norm, u_s_ma, 'r-', lw=2, label=r'$u_s^{Ma}$ (Marangoni surface)')
            ax.plot(self.r_norm, u_s, 'g--', lw=2, label=r'$u_s$ (surface total)')
            ax.plot(self.r_norm, u_b, 'm--', lw=2, label=r'$u_b$ (bulk total)')
            ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel(r'$r/R$', fontsize=12)
        ax.set_ylabel(r'Velocity', fontsize=12)
        ax.set_title(f'Velocity Components (t = {hist["t"][idx]:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_summary(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        self.plot_marangoni_effect(axes[0, 0])
        self.plot_velocity_comparison(axes[0, 1])
        self.plot_concentration_layers(axes[0, 2])
        self.plot_deposition_evolution(axes[1, 0])
        self.plot_final_deposition(axes[1, 1])
        self.plot_2d_pattern(axes[1, 2])
        plt.tight_layout()
        return fig


def create_animation_gif(sim, filename='coffee_ring_marangoni.gif', fps=10):
    hist = sim.history
    R = sim.p.R
    n_frames = len(hist['t'])
    n_grid = 100
    x = np.linspace(-R, R, n_grid)
    y = np.linspace(-R, R, n_grid)
    X, Y = np.meshgrid(x, y)
    R_grid = np.sqrt(X**2 + Y**2)
    frames = []
    print(f"  Generating {n_frames} frames...")
    for i in range(n_frames):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
        ax1 = axes[0]
        ax1.plot(sim.r/R, hist['u_s'][i], 'g-', lw=2, label='Surface')
        ax1.plot(sim.r/R, hist['u_b'][i], 'm--', lw=2, label='Bulk')
        ax1.axhline(0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlim(0, 1)
        u_max = max(np.max(np.abs(hist['u_s'][i])), np.max(np.abs(hist['u_b'][i])), 0.1)
        ax1.set_ylim(-u_max*1.2, u_max*1.2)
        ax1.set_xlabel(r'$r/R$', fontsize=11)
        ax1.set_ylabel('Velocity', fontsize=11)
        ax1.set_title('Layer Velocities', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax2 = axes[1]
        ax2.fill_between(sim.r/R, 0, hist['sigma'][i], alpha=0.6, color='saddlebrown')
        ax2.plot(sim.r/R, hist['sigma'][i], 'k-', lw=2)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, np.max(hist['sigma'][-1]) * 1.1 + 0.01)
        ax2.set_xlabel(r'$r/R$', fontsize=11)
        ax2.set_ylabel(r'Deposition $\sigma$', fontsize=11)
        ax2.set_title('Radial Profile', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax3 = axes[2]
        Z = np.interp(R_grid.flatten(), sim.r, hist['sigma'][i])
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > R] = np.nan
        im = ax3.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrBr', 
                        vmin=0, vmax=np.max(hist['sigma'][-1]) + 0.01)
        circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
        ax3.add_patch(circle)
        ax3.set_xlabel(r'$x/R$', fontsize=11)
        ax3.set_ylabel(r'$y/R$', fontsize=11)
        ax3.set_title('Top View', fontsize=11)
        ax3.set_aspect('equal')
        plt.colorbar(im, ax=ax3, shrink=0.8)
        progress = hist['t'][i] / sim.t_f
        fig.suptitle(f'Coffee Ring with Ma = {sim.p.Ma} - {100*progress:.0f}% Complete', 
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        fig.canvas.draw()
        try:
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)[:, :, :3]
        except AttributeError:
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
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(filename, save_all=True, append_images=imgs[1:], duration=1000//fps, loop=0)
        print(f"  Saved animation: {filename}")
    except ImportError:
        print("  PIL not available, skipping GIF save")


def compare_marangoni(ma_values, save_prefix='ma_comparison'):
    results = []
    for ma in ma_values:
        print(f"\n--- Running Ma = {ma} ---")
        params = Params(Ma=ma, Nr=200, t_end_frac=0.95)
        sim = CoffeeRingMarangoni(params)
        sim.run(verbose=False, save_interval=100)
        n = len(sim.sigma)
        center_dep = np.mean(sim.sigma[:n//4])
        edge_dep = np.mean(sim.sigma[-n//4:])
        peak_loc = sim.r[np.argmax(sim.sigma)] / sim.p.R
        results.append({
            'Ma': ma,
            'sim': sim,
            'center': center_dep,
            'edge': edge_dep,
            'ratio': edge_dep / (center_dep + 1e-10),
            'peak_loc': peak_loc,
            'peak_val': sim.sigma.max()
        })
    n_ma = len(ma_values)
    fig, axes = plt.subplots(2, min(n_ma, 4), figsize=(4*min(n_ma, 4), 8))
    if n_ma == 1:
        axes = axes.reshape(2, 1)
    for i, res in enumerate(results[:4]):
        ax_top = axes[0, i]
        ax_bot = axes[1, i]
        sim = res['sim']
        r_norm = sim.r / sim.p.R
        ax_top.fill_between(r_norm, 0, sim.sigma, alpha=0.5, color='saddlebrown')
        ax_top.plot(r_norm, sim.sigma, 'k-', lw=2)
        ax_top.set_xlim(0, 1)
        ax_top.set_xlabel(r'$r/R$')
        ax_top.set_ylabel(r'$\sigma$')
        ax_top.set_title(f'Ma = {res["Ma"]}\nRatio = {res["ratio"]:.1f}')
        ax_top.grid(True, alpha=0.3)
        R = sim.p.R
        n_grid = 100
        x = np.linspace(-R, R, n_grid)
        y = np.linspace(-R, R, n_grid)
        X, Y = np.meshgrid(x, y)
        R_grid = np.sqrt(X**2 + Y**2)
        Z = np.interp(R_grid.flatten(), sim.r, sim.sigma)
        Z = Z.reshape(R_grid.shape)
        Z[R_grid > R] = np.nan
        ax_bot.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='YlOrBr')
        circle = plt.Circle((0, 0), 1, fill=False, color='black', lw=2)
        ax_bot.add_patch(circle)
        ax_bot.set_aspect('equal')
        ax_bot.set_xlabel(r'$x/R$')
        ax_bot.set_ylabel(r'$y/R$')
    plt.tight_layout()
    fig.savefig(f'{save_prefix}_patterns.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {save_prefix}_patterns.png")
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    ma_arr = [r['Ma'] for r in results]
    ratio_arr = [r['ratio'] for r in results]
    peak_arr = [r['peak_loc'] for r in results]
    axes2[0].plot(ma_arr, ratio_arr, 'ro-', markersize=10, lw=2)
    axes2[0].set_xlabel('Marangoni Number', fontsize=12)
    axes2[0].set_ylabel('Edge/Center Ratio', fontsize=12)
    axes2[0].set_title('Ring Enhancement vs Ma', fontsize=13, fontweight='bold')
    axes2[0].grid(True, alpha=0.3)
    if max(ma_arr) > 100:
        axes2[0].set_xscale('symlog', linthresh=10)
    axes2[1].plot(ma_arr, peak_arr, 'bs-', markersize=10, lw=2)
    axes2[1].set_xlabel('Marangoni Number', fontsize=12)
    axes2[1].set_ylabel('Peak Location (r/R)', fontsize=12)
    axes2[1].set_title('Deposit Peak Location vs Ma', fontsize=13, fontweight='bold')
    axes2[1].set_ylim(0, 1.05)
    axes2[1].grid(True, alpha=0.3)
    if max(ma_arr) > 100:
        axes2[1].set_xscale('symlog', linthresh=10)
    plt.tight_layout()
    fig2.savefig(f'{save_prefix}_metrics.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {save_prefix}_metrics.png")
    return results


if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'compare':
        print("="*60)
        print("MARANGONI NUMBER COMPARISON")
        print("="*60)
        ma_values = [0, 0.5, 1.0, 2.0, 5.0, 10.0]
        results = compare_marangoni(ma_values)
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"{'Ma':>8} {'Edge/Center':>12} {'Peak r/R':>10} {'Peak sigma':>10}")
        print("-"*44)
        for r in results:
            print(f"{r['Ma']:>8.1f} {r['ratio']:>12.2f} {r['peak_loc']:>10.3f} {r['peak_val']:>10.3f}")
    else:
        ma = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
        print("="*60)
        print(f"SINGLE SIMULATION: Ma = {ma}")
        print("="*60)
        params = Params(Ma=ma, Nr=200, t_end_frac=0.95)
        sim = CoffeeRingMarangoni(params)
        sim.run(verbose=True, save_interval=20)
        print("\nGenerating plots...")
        viz = Visualizer(sim)
        fig = viz.plot_summary()
        fig.savefig('coffee_ring_marangoni_summary.png', dpi=150, bbox_inches='tight')
        print("  Saved: coffee_ring_marangoni_summary.png")
        print("\nGenerating animation...")
        create_animation_gif(sim, 'coffee_ring_marangoni.gif', fps=10)
        print("\n" + "="*60)
        print("STATISTICS")
        print("="*60)
        n = len(sim.sigma)
        center_dep = np.mean(sim.sigma[:n//4])
        edge_dep = np.mean(sim.sigma[-n//4:])
        print(f"  Ma = {ma}")
        print(f"  Center deposition: {center_dep:.4f}")
        print(f"  Edge deposition:   {edge_dep:.4f}")
        print(f"  Edge/Center ratio: {edge_dep/(center_dep+1e-10):.2f}")
        print(f"  Peak location:     {sim.r[np.argmax(sim.sigma)]/sim.p.R:.3f}")
        print("="*60)
