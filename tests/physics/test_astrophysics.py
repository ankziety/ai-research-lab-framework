"""
Astrophysics Tests

Comprehensive unit tests for astrophysics simulations and calculations.
Tests stellar evolution, cosmology, galactic dynamics, and N-body simulations.
"""

import pytest
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Any, Optional
from unittest.mock import Mock, patch

# Astrophysics test markers
pytestmark = pytest.mark.astrophysics


class MockAstrophysicsAgent:
    """Mock astrophysics agent for testing."""
    
    def __init__(self):
        self.name = "AstrophysicsAgent"
        self.initialized = True
        self.stellar_database = {}
        self.cosmological_parameters = {
            'H0': 70.0,  # km/s/Mpc
            'Omega_m': 0.3,
            'Omega_lambda': 0.7,
            'Omega_b': 0.05
        }
    
    def simulate_stellar_evolution(self, initial_mass: float, metallicity: float = 0.02,
                                 age_range: Tuple[float, float] = (0, 1e10)) -> Dict[str, Any]:
        """Simulate stellar evolution over time."""
        M_sun = 1.989e30  # kg
        mass_solar_units = initial_mass
        
        # Time evolution
        t_start, t_end = age_range
        n_points = 1000
        times = np.linspace(t_start, t_end, n_points)
        
        # Main sequence lifetime (simplified mass-luminosity relation)
        if mass_solar_units <= 0.5:
            ms_lifetime = 5.6e10  # years
        elif mass_solar_units <= 25:
            ms_lifetime = 1e10 * (mass_solar_units)**(-2.5)
        else:
            ms_lifetime = 3.2e6 * (mass_solar_units)**(-2.5)
        
        # Stellar properties evolution
        luminosities = np.zeros(n_points)
        temperatures = np.zeros(n_points)
        radii = np.zeros(n_points)
        phases = []
        
        for i, t in enumerate(times):
            if t < ms_lifetime:
                # Main sequence
                phase = 'main_sequence'
                # Mass-luminosity relation: L ∝ M^3.5
                L_ms = mass_solar_units**3.5
                luminosities[i] = L_ms
                
                # Mass-temperature relation
                T_eff = 5778 * (mass_solar_units**0.5)  # K
                temperatures[i] = T_eff
                
                # Stefan-Boltzmann law for radius
                sigma_sb = 5.67e-8  # W/m²/K⁴
                L_sun = 3.828e26  # W
                R_sun = 6.96e8   # m
                
                R = R_sun * np.sqrt(L_ms) / (T_eff/5778)**2
                radii[i] = R / R_sun  # Solar radii
                
            elif mass_solar_units < 8:
                # Low-mass star evolution: Red Giant → White Dwarf
                if t < ms_lifetime * 1.1:
                    phase = 'red_giant'
                    luminosities[i] = mass_solar_units**3.5 * 100  # Luminous giant
                    temperatures[i] = 3500  # Cool giant
                    radii[i] = 50  # Large radius
                else:
                    phase = 'white_dwarf'
                    luminosities[i] = 0.01  # Dim white dwarf
                    temperatures[i] = 10000  # Hot but small
                    radii[i] = 0.01  # Earth-sized
                    
            else:
                # High-mass star evolution: Supergiant → Supernova → Neutron Star/Black Hole
                if t < ms_lifetime * 1.05:
                    phase = 'supergiant'
                    luminosities[i] = mass_solar_units**3.5 * 1000
                    temperatures[i] = 4000
                    radii[i] = 200
                else:
                    # Post-supernova remnant
                    if mass_solar_units < 20:
                        phase = 'neutron_star'
                        luminosities[i] = 1e-6
                        temperatures[i] = 1e6
                        radii[i] = 1e-5  # ~10 km
                    else:
                        phase = 'black_hole'
                        luminosities[i] = 0
                        temperatures[i] = 0
                        radii[i] = 2 * mass_solar_units * 2.95e3 / 6.96e8  # Schwarzschild radius
            
            phases.append(phase)
        
        return {
            'times': times,
            'luminosities': luminosities,
            'temperatures': temperatures,
            'radii': radii,
            'phases': phases,
            'initial_mass': initial_mass,
            'metallicity': metallicity,
            'main_sequence_lifetime': ms_lifetime,
            'final_phase': phases[-1],
            'peak_luminosity': np.max(luminosities)
        }
    
    def calculate_orbital_mechanics(self, body1_mass: float, body2_mass: float,
                                  semi_major_axis: float) -> Dict[str, Any]:
        """Calculate orbital mechanics for two-body system."""
        G = 6.674e-11  # m³/kg/s²
        
        # Total mass
        total_mass = body1_mass + body2_mass
        
        # Orbital period (Kepler's third law)
        period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * total_mass))
        
        # Orbital velocity (circular orbit approximation)
        orbital_velocity = np.sqrt(G * total_mass / semi_major_axis)
        
        # Reduced mass
        reduced_mass = (body1_mass * body2_mass) / total_mass
        
        # Center of mass distances
        r1 = semi_major_axis * body2_mass / total_mass
        r2 = semi_major_axis * body1_mass / total_mass
        
        # Escape velocity
        escape_velocity = np.sqrt(2 * G * total_mass / semi_major_axis)
        
        # Hill sphere (simplified)
        hill_radius = semi_major_axis * (reduced_mass / (3 * total_mass))**(1/3)
        
        return {
            'orbital_period': period,
            'orbital_velocity': orbital_velocity,
            'escape_velocity': escape_velocity,
            'reduced_mass': reduced_mass,
            'center_of_mass_distances': (r1, r2),
            'hill_radius': hill_radius,
            'semi_major_axis': semi_major_axis,
            'gravitational_binding_energy': -G * body1_mass * body2_mass / semi_major_axis
        }
    
    def simulate_galaxy_rotation_curve(self, galaxy_mass: float, dark_matter_fraction: float = 0.85,
                                     max_radius: float = 50000) -> Dict[str, Any]:
        """Simulate galaxy rotation curve including dark matter."""
        # Radial distances (parsecs)
        radii = np.linspace(100, max_radius, 1000)
        
        # Stellar mass distribution (exponential disk)
        scale_length = max_radius * 0.2
        stellar_mass_profile = np.exp(-radii / scale_length)
        stellar_mass_profile /= np.trapz(stellar_mass_profile, radii)
        stellar_mass_profile *= galaxy_mass * (1 - dark_matter_fraction)
        
        # Dark matter halo (NFW profile simplified)
        dark_matter_mass = galaxy_mass * dark_matter_fraction
        halo_scale_radius = max_radius * 0.1
        
        # Enclosed mass calculation
        G = 6.674e-11  # m³/kg/s²
        pc_to_m = 3.086e16  # parsecs to meters
        
        enclosed_stellar_mass = np.zeros_like(radii)
        enclosed_dm_mass = np.zeros_like(radii)
        
        for i, r in enumerate(radii):
            # Stellar mass within radius r
            mask = radii <= r
            enclosed_stellar_mass[i] = np.trapz(stellar_mass_profile[mask], radii[mask])
            
            # Dark matter mass (simplified NFW)
            enclosed_dm_mass[i] = dark_matter_mass * (r / (r + halo_scale_radius))
        
        total_enclosed_mass = enclosed_stellar_mass + enclosed_dm_mass
        
        # Rotation velocities
        rotation_velocities = np.sqrt(G * total_enclosed_mass * 1.989e30 / (radii * pc_to_m))
        
        # Stellar component only (for comparison)
        stellar_velocities = np.sqrt(G * enclosed_stellar_mass * 1.989e30 / (radii * pc_to_m))
        
        return {
            'radii': radii,  # parsecs
            'rotation_velocities': rotation_velocities,  # m/s
            'stellar_velocities': stellar_velocities,
            'enclosed_stellar_mass': enclosed_stellar_mass,  # solar masses
            'enclosed_dark_matter_mass': enclosed_dm_mass,
            'total_enclosed_mass': total_enclosed_mass,
            'dark_matter_fraction': dark_matter_fraction,
            'peak_velocity': np.max(rotation_velocities),
            'flat_rotation_evidence': np.std(rotation_velocities[-100:]) / np.mean(rotation_velocities[-100:]) < 0.1
        }
    
    def calculate_cosmological_distances(self, redshift: float) -> Dict[str, Any]:
        """Calculate cosmological distances for given redshift."""
        # Cosmological parameters
        H0 = self.cosmological_parameters['H0']  # km/s/Mpc
        Omega_m = self.cosmological_parameters['Omega_m']
        Omega_lambda = self.cosmological_parameters['Omega_lambda']
        
        c = 2.998e5  # km/s
        
        # Hubble distance
        d_H = c / H0  # Mpc
        
        # Comoving distance (flat universe approximation)
        def E(z):
            return np.sqrt(Omega_m * (1 + z)**3 + Omega_lambda)
        
        # Numerical integration for comoving distance
        z_array = np.linspace(0, redshift, 1000)
        integrand = 1 / E(z_array)
        comoving_distance = d_H * np.trapz(integrand, z_array)
        
        # Angular diameter distance
        angular_diameter_distance = comoving_distance / (1 + redshift)
        
        # Luminosity distance
        luminosity_distance = comoving_distance * (1 + redshift)
        
        # Light travel time
        def time_integrand(z):
            return 1 / ((1 + z) * E(z))
        
        time_array = np.linspace(0, redshift, 1000)
        time_integrand_values = time_integrand(time_array)
        light_travel_time = (d_H / H0) * np.trapz(time_integrand_values, time_array) * 977.8  # Gyr
        
        # Age of universe at redshift z
        age_at_z = 13.8 - light_travel_time  # Approximate
        
        return {
            'redshift': redshift,
            'comoving_distance': comoving_distance,  # Mpc
            'angular_diameter_distance': angular_diameter_distance,
            'luminosity_distance': luminosity_distance,
            'light_travel_time': light_travel_time,  # Gyr
            'age_at_redshift': age_at_z,
            'distance_modulus': 5 * np.log10(luminosity_distance * 1e6 / 10),
            'scale_factor': 1 / (1 + redshift)
        }
    
    def simulate_supernova_light_curve(self, supernova_type: str = 'Ia',
                                     peak_magnitude: float = -19.3) -> Dict[str, Any]:
        """Simulate supernova light curve."""
        # Time array (days relative to peak)
        times = np.linspace(-20, 100, 300)
        
        if supernova_type == 'Ia':
            # Type Ia supernova template
            rise_time = 19  # days to peak
            decline_rate = 0.98  # mag per 15 days after peak
            
            magnitudes = np.zeros_like(times)
            
            for i, t in enumerate(times):
                if t <= 0:
                    # Pre-peak rise (exponential)
                    magnitudes[i] = peak_magnitude + 5 * np.exp(t / rise_time) - 5
                else:
                    # Post-peak decline
                    if t <= 15:
                        # Early decline
                        magnitudes[i] = peak_magnitude + decline_rate * (t / 15)
                    else:
                        # Late decline (faster)
                        magnitudes[i] = peak_magnitude + decline_rate + 2 * ((t - 15) / 30)
        
        elif supernova_type == 'II':
            # Type II supernova (core collapse)
            rise_time = 10
            plateau_duration = 80
            plateau_magnitude = peak_magnitude + 1.5
            
            magnitudes = np.zeros_like(times)
            
            for i, t in enumerate(times):
                if t <= 0:
                    magnitudes[i] = peak_magnitude + 3 * np.exp(t / rise_time) - 3
                elif t <= plateau_duration:
                    magnitudes[i] = plateau_magnitude
                else:
                    magnitudes[i] = plateau_magnitude + 0.5 * ((t - plateau_duration) / 20)
        
        else:
            # Generic exponential decline
            magnitudes = peak_magnitude + 0.1 * times
        
        # Convert to flux
        flux = 10**(-0.4 * (magnitudes - peak_magnitude))
        
        # Calculate key parameters
        peak_flux = np.max(flux)
        time_to_peak = times[np.argmax(flux)]
        magnitude_15_days = np.interp(15, times, magnitudes)
        decline_rate_15 = magnitude_15_days - peak_magnitude
        
        return {
            'times': times,  # days
            'magnitudes': magnitudes,
            'flux': flux,
            'supernova_type': supernova_type,
            'peak_magnitude': peak_magnitude,
            'peak_flux': peak_flux,
            'time_to_peak': time_to_peak,
            'decline_rate_15_days': decline_rate_15,
            'total_energy': np.trapz(flux, times) * 86400  # integrated flux
        }
    
    def simulate_nbody_system(self, n_bodies: int, box_size: float = 1.0,
                            n_steps: int = 1000, dt: float = 0.001) -> Dict[str, Any]:
        """Simulate N-body gravitational system."""
        # Initialize particles
        np.random.seed(42)  # For reproducible tests
        positions = np.random.uniform(-box_size/2, box_size/2, (n_bodies, 3))
        velocities = np.random.normal(0, 0.1, (n_bodies, 3))
        masses = np.random.uniform(0.5, 2.0, n_bodies)
        
        # Storage for trajectory
        trajectory = {
            'positions': np.zeros((n_steps, n_bodies, 3)),
            'velocities': np.zeros((n_steps, n_bodies, 3)),
            'kinetic_energy': np.zeros(n_steps),
            'potential_energy': np.zeros(n_steps),
            'total_energy': np.zeros(n_steps)
        }
        
        # Simulation parameters
        G = 1.0  # Gravitational constant (units chosen for convenience)
        softening = 0.01  # Softening length to avoid singularities
        
        for step in range(n_steps):
            # Calculate forces
            forces = np.zeros_like(positions)
            potential_energy = 0.0
            
            for i in range(n_bodies):
                for j in range(i + 1, n_bodies):
                    # Distance vector
                    dr = positions[j] - positions[i]
                    r = np.linalg.norm(dr)
                    
                    if r > softening:
                        # Gravitational force
                        F_mag = G * masses[i] * masses[j] / (r**2 + softening**2)
                        F_vec = F_mag * dr / r
                        
                        forces[i] += F_vec
                        forces[j] -= F_vec
                        
                        # Potential energy
                        potential_energy -= G * masses[i] * masses[j] / r
            
            # Leapfrog integration
            velocities += forces / masses[:, np.newaxis] * dt
            positions += velocities * dt
            
            # Calculate energies
            kinetic_energy = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2)
            total_energy = kinetic_energy + potential_energy
            
            # Store trajectory
            trajectory['positions'][step] = positions.copy()
            trajectory['velocities'][step] = velocities.copy()
            trajectory['kinetic_energy'][step] = kinetic_energy
            trajectory['potential_energy'][step] = potential_energy
            trajectory['total_energy'][step] = total_energy
        
        # Calculate system properties
        center_of_mass = np.mean(positions, axis=0)
        total_momentum = np.sum(masses[:, np.newaxis] * velocities, axis=0)
        virial_ratio = 2 * trajectory['kinetic_energy'][-1] / abs(trajectory['potential_energy'][-1])
        
        # Energy conservation check
        energy_conservation = abs(trajectory['total_energy'][-1] - trajectory['total_energy'][0]) / abs(trajectory['total_energy'][0])
        
        return {
            'trajectory': trajectory,
            'final_positions': positions,
            'final_velocities': velocities,
            'masses': masses,
            'center_of_mass': center_of_mass,
            'total_momentum': total_momentum,
            'virial_ratio': virial_ratio,
            'energy_conservation': energy_conservation,
            'simulation_parameters': {
                'n_bodies': n_bodies,
                'n_steps': n_steps,
                'dt': dt,
                'box_size': box_size
            }
        }
    
    def calculate_black_hole_properties(self, mass: float) -> Dict[str, Any]:
        """Calculate black hole properties."""
        M_sun = 1.989e30  # kg
        mass_kg = mass * M_sun
        
        # Physical constants
        G = 6.674e-11  # m³/kg/s²
        c = 2.998e8   # m/s
        h = 6.626e-34 # J·s
        k_B = 1.381e-23  # J/K
        
        # Schwarzschild radius
        r_s = 2 * G * mass_kg / c**2
        
        # Event horizon area
        area = 4 * np.pi * r_s**2
        
        # Hawking temperature
        T_hawking = h * c**3 / (8 * np.pi * G * mass_kg * k_B)
        
        # Hawking luminosity
        L_hawking = h * c**6 / (15360 * np.pi * G**2 * mass_kg**2)
        
        # Evaporation time
        t_evaporation = 5120 * np.pi * G**2 * mass_kg**3 / (h * c**4)
        
        # Tidal acceleration at event horizon
        tidal_acceleration = G * mass_kg / r_s**3
        
        return {
            'mass': mass,  # solar masses
            'schwarzschild_radius': r_s,  # meters
            'event_horizon_area': area,   # m²
            'hawking_temperature': T_hawking,  # K
            'hawking_luminosity': L_hawking,   # W
            'evaporation_time': t_evaporation, # seconds
            'tidal_acceleration': tidal_acceleration,  # m/s²
            'is_stellar_mass': 3 < mass < 100,
            'is_supermassive': mass > 1e6,
            'photon_sphere_radius': 1.5 * r_s
        }
    
    def simulate_cosmic_microwave_background(self, l_max: int = 2500) -> Dict[str, Any]:
        """Simulate cosmic microwave background power spectrum."""
        # Multipole moments
        ell = np.arange(2, l_max + 1)
        
        # Mock CMB power spectrum with realistic features
        # First acoustic peak around l ~ 220
        # Second peak around l ~ 540
        # Third peak around l ~ 800
        
        # Base power spectrum (simplified model)
        C_l = np.zeros_like(ell, dtype=float)
        
        # First acoustic peak
        peak1_center = 220
        peak1_amplitude = 5500
        peak1_width = 50
        C_l += peak1_amplitude * np.exp(-0.5 * ((ell - peak1_center) / peak1_width)**2)
        
        # Second acoustic peak
        peak2_center = 540
        peak2_amplitude = 2500
        peak2_width = 80
        C_l += peak2_amplitude * np.exp(-0.5 * ((ell - peak2_center) / peak2_width)**2)
        
        # Third acoustic peak
        peak3_center = 800
        peak3_amplitude = 1500
        peak3_width = 100
        C_l += peak3_amplitude * np.exp(-0.5 * ((ell - peak3_center) / peak3_width)**2)
        
        # Damping tail at high l
        damping = np.exp(-ell / 1000)
        C_l *= damping
        
        # Add ISW plateau at low l
        isw_plateau = 1000 * (ell / 100)**(-2)
        C_l += isw_plateau
        
        # Scale to microKelvin squared
        C_l *= (2.725e6)**2  # CMB temperature in microKelvin
        
        # Calculate derived parameters
        acoustic_scale = 2 * np.pi / peak1_center  # radians
        sound_horizon = acoustic_scale * 14000  # Mpc (approximate)
        
        return {
            'multipoles': ell,
            'power_spectrum': C_l,  # μK²
            'first_acoustic_peak': peak1_center,
            'acoustic_scale': acoustic_scale,
            'sound_horizon': sound_horizon,
            'integrated_power': np.trapz(C_l, ell),
            'peak_positions': [peak1_center, peak2_center, peak3_center],
            'cmb_temperature': 2.725  # K
        }


class TestAstrophysicsAgent:
    """Test class for astrophysics functionality."""
    
    @pytest.fixture
    def astrophysics_agent(self):
        """Create an astrophysics agent instance for testing."""
        return MockAstrophysicsAgent()
    
    def test_agent_initialization(self, astrophysics_agent):
        """Test astrophysics agent initialization."""
        assert astrophysics_agent.name == "AstrophysicsAgent"
        assert astrophysics_agent.initialized is True
        assert hasattr(astrophysics_agent, 'stellar_database')
        assert hasattr(astrophysics_agent, 'cosmological_parameters')
        
        # Check cosmological parameters
        cosmo_params = astrophysics_agent.cosmological_parameters
        assert 'H0' in cosmo_params
        assert 'Omega_m' in cosmo_params
        assert 'Omega_lambda' in cosmo_params
        assert cosmo_params['H0'] > 0
    
    def test_stellar_evolution_simulation(self, astrophysics_agent):
        """Test stellar evolution simulation."""
        # Test solar-mass star
        initial_mass = 1.0  # Solar masses
        metallicity = 0.02
        age_range = (0, 1e10)  # 10 Gyr
        
        result = astrophysics_agent.simulate_stellar_evolution(initial_mass, metallicity, age_range)
        
        assert 'times' in result
        assert 'luminosities' in result
        assert 'temperatures' in result
        assert 'radii' in result
        assert 'phases' in result
        assert 'main_sequence_lifetime' in result
        
        # Check data consistency
        assert len(result['times']) == len(result['luminosities'])
        assert len(result['times']) == len(result['temperatures'])
        assert len(result['times']) == len(result['radii'])
        
        # Check physical constraints
        assert all(L > 0 for L in result['luminosities'])
        assert all(T > 0 for T in result['temperatures'])
        assert all(R > 0 for R in result['radii'])
        
        # Check main sequence lifetime is reasonable for solar mass star
        ms_lifetime = result['main_sequence_lifetime']
        assert 5e9 < ms_lifetime < 2e10  # Should be around 10 Gyr
    
    @pytest.mark.parametrize("stellar_mass", [0.5, 1.0, 5.0, 15.0])
    def test_stellar_evolution_mass_dependence(self, astrophysics_agent, stellar_mass):
        """Test stellar evolution for different masses."""
        result = astrophysics_agent.simulate_stellar_evolution(stellar_mass)
        
        # More massive stars should have shorter lifetimes
        ms_lifetime = result['main_sequence_lifetime']
        
        if stellar_mass > 1.0:
            assert ms_lifetime < 1e10  # Shorter than solar lifetime
        elif stellar_mass < 1.0:
            assert ms_lifetime > 1e10  # Longer than solar lifetime
        
        # Check final evolutionary phase
        final_phase = result['final_phase']
        
        if stellar_mass < 8:
            assert final_phase in ['main_sequence', 'red_giant', 'white_dwarf']
        else:
            assert final_phase in ['supergiant', 'neutron_star', 'black_hole']
    
    def test_orbital_mechanics_calculation(self, astrophysics_agent):
        """Test orbital mechanics calculations."""
        # Earth-Sun system
        earth_mass = 5.972e24  # kg
        sun_mass = 1.989e30    # kg
        earth_orbit = 1.496e11 # meters (1 AU)
        
        result = astrophysics_agent.calculate_orbital_mechanics(sun_mass, earth_mass, earth_orbit)
        
        assert 'orbital_period' in result
        assert 'orbital_velocity' in result
        assert 'escape_velocity' in result
        assert 'gravitational_binding_energy' in result
        
        # Check orbital period (should be close to 1 year)
        period_years = result['orbital_period'] / (365.25 * 24 * 3600)
        assert abs(period_years - 1.0) < 0.01  # Within 1%
        
        # Check orbital velocity (should be ~30 km/s)
        orbital_vel_km_s = result['orbital_velocity'] / 1000
        assert 25 < orbital_vel_km_s < 35
        
        # Check that escape velocity > orbital velocity
        escape_vel = result['escape_velocity']
        orbital_vel = result['orbital_velocity']
        assert escape_vel > orbital_vel
        
        # Check binding energy is negative
        assert result['gravitational_binding_energy'] < 0
    
    def test_galaxy_rotation_curve_simulation(self, astrophysics_agent):
        """Test galaxy rotation curve simulation."""
        galaxy_mass = 1e12  # Solar masses (Milky Way-like)
        dark_matter_fraction = 0.85
        max_radius = 30000  # parsecs
        
        result = astrophysics_agent.simulate_galaxy_rotation_curve(
            galaxy_mass, dark_matter_fraction, max_radius
        )
        
        assert 'radii' in result
        assert 'rotation_velocities' in result
        assert 'stellar_velocities' in result
        assert 'enclosed_dark_matter_mass' in result
        assert 'flat_rotation_evidence' in result
        
        radii = result['radii']
        rot_vel = result['rotation_velocities']
        stellar_vel = result['stellar_velocities']
        
        # Check that rotation velocities are reasonable (100-300 km/s)
        rot_vel_km_s = rot_vel / 1000
        assert all(50 < v < 500 for v in rot_vel_km_s)
        
        # Dark matter should cause flatter rotation curve than stellar alone
        # Outer regions should show the difference
        outer_indices = radii > max_radius * 0.8
        if np.any(outer_indices):
            outer_rot_vel = np.mean(rot_vel[outer_indices])
            outer_stellar_vel = np.mean(stellar_vel[outer_indices])
            assert outer_rot_vel > outer_stellar_vel
        
        # Check dark matter fraction
        total_dm = result['enclosed_dark_matter_mass'][-1]
        total_stellar = result['enclosed_stellar_mass'][-1]
        actual_dm_fraction = total_dm / (total_dm + total_stellar)
        assert abs(actual_dm_fraction - dark_matter_fraction) < 0.1
    
    def test_cosmological_distances_calculation(self, astrophysics_agent):
        """Test cosmological distance calculations."""
        redshift = 1.0  # z = 1
        
        result = astrophysics_agent.calculate_cosmological_distances(redshift)
        
        assert 'redshift' in result
        assert 'comoving_distance' in result
        assert 'angular_diameter_distance' in result
        assert 'luminosity_distance' in result
        assert 'light_travel_time' in result
        assert 'distance_modulus' in result
        
        # Check distance relationships
        comoving_dist = result['comoving_distance']
        angular_dist = result['angular_diameter_distance']
        luminosity_dist = result['luminosity_distance']
        
        # For z = 1: d_A = d_C / (1+z), d_L = d_C * (1+z)
        expected_angular = comoving_dist / (1 + redshift)
        expected_luminosity = comoving_dist * (1 + redshift)
        
        assert abs(angular_dist - expected_angular) < expected_angular * 0.01
        assert abs(luminosity_dist - expected_luminosity) < expected_luminosity * 0.01
        
        # Light travel time should be reasonable
        light_time = result['light_travel_time']
        assert 0 < light_time < 13.8  # Less than age of universe
        
        # Distance modulus should be reasonable
        dist_mod = result['distance_modulus']
        assert 40 < dist_mod < 50  # Typical for z=1 objects
    
    @pytest.mark.parametrize("sn_type", ['Ia', 'II'])
    def test_supernova_light_curve_simulation(self, astrophysics_agent, sn_type):
        """Test supernova light curve simulation."""
        peak_magnitude = -19.0
        
        result = astrophysics_agent.simulate_supernova_light_curve(sn_type, peak_magnitude)
        
        assert 'times' in result
        assert 'magnitudes' in result
        assert 'flux' in result
        assert 'supernova_type' in result
        assert 'decline_rate_15_days' in result
        
        times = result['times']
        magnitudes = result['magnitudes']
        flux = result['flux']
        
        # Check that peak magnitude is achieved
        min_magnitude = np.min(magnitudes)
        assert abs(min_magnitude - peak_magnitude) < 0.1
        
        # Check that flux and magnitude are inversely related
        peak_flux_idx = np.argmax(flux)
        min_mag_idx = np.argmin(magnitudes)
        assert abs(peak_flux_idx - min_mag_idx) <= 1  # Should be same or adjacent
        
        # Type-specific checks
        if sn_type == 'Ia':
            # Type Ia should have specific decline rate
            decline_rate = result['decline_rate_15_days']
            assert 0.5 < decline_rate < 2.0  # Typical range
        
        elif sn_type == 'II':
            # Type II should show plateau behavior
            # Check for relatively constant magnitude over some period
            plateau_mask = (times > 10) & (times < 60)
            if np.any(plateau_mask):
                plateau_mags = magnitudes[plateau_mask]
                plateau_variation = np.std(plateau_mags)
                assert plateau_variation < 1.0  # Less than 1 mag variation
    
    def test_nbody_system_simulation(self, astrophysics_agent):
        """Test N-body gravitational simulation."""
        n_bodies = 10
        box_size = 2.0
        n_steps = 100
        dt = 0.01
        
        result = astrophysics_agent.simulate_nbody_system(n_bodies, box_size, n_steps, dt)
        
        assert 'trajectory' in result
        assert 'final_positions' in result
        assert 'final_velocities' in result
        assert 'energy_conservation' in result
        assert 'virial_ratio' in result
        
        trajectory = result['trajectory']
        
        # Check trajectory dimensions
        assert trajectory['positions'].shape == (n_steps, n_bodies, 3)
        assert trajectory['velocities'].shape == (n_steps, n_bodies, 3)
        assert len(trajectory['total_energy']) == n_steps
        
        # Check energy conservation (should be reasonable for short simulation)
        energy_conservation = result['energy_conservation']
        assert energy_conservation < 0.1  # Within 10%
        
        # Check momentum conservation (total momentum should be conserved)
        total_momentum = result['total_momentum']
        momentum_magnitude = np.linalg.norm(total_momentum)
        assert momentum_magnitude < 0.1  # Should be small for random initial conditions
        
        # Check virial ratio (should approach 0.5 for bound system)
        virial_ratio = result['virial_ratio']
        assert virial_ratio > 0  # Should be positive
    
    def test_black_hole_properties_calculation(self, astrophysics_agent):
        """Test black hole properties calculation."""
        # Test stellar mass black hole
        mass = 10.0  # Solar masses
        
        result = astrophysics_agent.calculate_black_hole_properties(mass)
        
        assert 'mass' in result
        assert 'schwarzschild_radius' in result
        assert 'hawking_temperature' in result
        assert 'evaporation_time' in result
        assert 'is_stellar_mass' in result
        
        # Check Schwarzschild radius scaling
        r_s = result['schwarzschild_radius']
        expected_r_s = 2 * 6.674e-11 * mass * 1.989e30 / (2.998e8)**2
        assert abs(r_s - expected_r_s) / expected_r_s < 1e-10
        
        # Check that more massive black holes are cooler
        mass_heavy = 100.0
        result_heavy = astrophysics_agent.calculate_black_hole_properties(mass_heavy)
        
        assert result_heavy['hawking_temperature'] < result['hawking_temperature']
        assert result_heavy['evaporation_time'] > result['evaporation_time']
        
        # Check classification
        assert result['is_stellar_mass'] is True
        assert result['is_supermassive'] is False
        
        # Test supermassive black hole
        mass_smbh = 1e9  # Solar masses
        result_smbh = astrophysics_agent.calculate_black_hole_properties(mass_smbh)
        assert result_smbh['is_supermassive'] is True
    
    def test_cosmic_microwave_background_simulation(self, astrophysics_agent):
        """Test cosmic microwave background simulation."""
        l_max = 1000
        
        result = astrophysics_agent.simulate_cosmic_microwave_background(l_max)
        
        assert 'multipoles' in result
        assert 'power_spectrum' in result
        assert 'first_acoustic_peak' in result
        assert 'acoustic_scale' in result
        assert 'peak_positions' in result
        
        ell = result['multipoles']
        C_l = result['power_spectrum']
        
        # Check multipoles range
        assert len(ell) == l_max - 1  # From 2 to l_max
        assert ell[0] == 2
        assert ell[-1] == l_max
        
        # Check power spectrum is positive
        assert all(C_l > 0)
        
        # Check first acoustic peak position
        first_peak = result['first_acoustic_peak']
        assert 200 < first_peak < 250  # Should be around 220
        
        # Check that peak is actually a peak (higher than neighboring values)
        peak_idx = np.argmin(np.abs(ell - first_peak))
        if peak_idx > 5 and peak_idx < len(C_l) - 5:
            peak_value = C_l[peak_idx]
            nearby_values = C_l[peak_idx-5:peak_idx+5]
            assert peak_value >= np.max(nearby_values) * 0.9  # Within 90% of local max
    
    @pytest.mark.slow
    def test_large_scale_nbody_simulation(self, astrophysics_agent):
        """Test N-body simulation with more particles."""
        n_bodies = 50
        n_steps = 500
        
        result = astrophysics_agent.simulate_nbody_system(n_bodies, n_steps=n_steps)
        
        # Should complete without errors
        assert result['simulation_parameters']['n_bodies'] == n_bodies
        assert len(result['final_positions']) == n_bodies
        
        # Energy conservation should still be reasonable
        assert result['energy_conservation'] < 0.2  # Within 20% for larger system
    
    def test_astrophysics_error_handling(self, astrophysics_agent):
        """Test error handling in astrophysics calculations."""
        # Test with invalid stellar mass
        result = astrophysics_agent.simulate_stellar_evolution(-1.0)
        assert 'main_sequence_lifetime' in result  # Should not crash
        
        # Test with zero mass orbital system
        with pytest.raises((ZeroDivisionError, ValueError)):
            astrophysics_agent.calculate_orbital_mechanics(0, 0, 1e11)
        
        # Test with negative redshift
        result = astrophysics_agent.calculate_cosmological_distances(-0.1)
        assert 'comoving_distance' in result  # Should handle gracefully
        
        # Test N-body with zero particles
        with pytest.raises((ValueError, IndexError)):
            astrophysics_agent.simulate_nbody_system(0)


class TestAstrophysicsIntegration:
    """Integration tests for astrophysics workflows."""
    
    @pytest.fixture
    def astrophysics_workflow(self):
        """Create an astrophysics workflow."""
        agent = MockAstrophysicsAgent()
        
        workflow = {
            'agent': agent,
            'observations': {},
            'simulations': {},
            'analysis_results': {}
        }
        
        return workflow
    
    def test_complete_stellar_system_analysis(self, astrophysics_workflow, physics_test_config):
        """Test complete stellar system analysis workflow."""
        agent = astrophysics_workflow['agent']
        
        # Step 1: Stellar evolution
        primary_mass = 2.0  # Solar masses
        stellar_evolution = agent.simulate_stellar_evolution(primary_mass, age_range=(0, 5e9))
        
        # Step 2: Binary orbital mechanics
        secondary_mass = 1.5  # Solar masses
        orbital_separation = 2e11  # meters (1.3 AU)
        orbital_mechanics = agent.calculate_orbital_mechanics(
            primary_mass * 1.989e30, secondary_mass * 1.989e30, orbital_separation
        )
        
        # Step 3: Potential supernova simulation
        if primary_mass > 8:  # Massive enough for supernova
            supernova_lc = agent.simulate_supernova_light_curve('II', peak_magnitude=-17.5)
        else:
            supernova_lc = None
        
        # Step 4: System black hole formation
        if primary_mass > 25:
            bh_properties = agent.calculate_black_hole_properties(primary_mass * 0.5)  # Remnant mass
        else:
            bh_properties = None
        
        # Store complete analysis
        astrophysics_workflow['analysis_results'] = {
            'stellar_evolution': stellar_evolution,
            'orbital_mechanics': orbital_mechanics,
            'supernova': supernova_lc,
            'black_hole': bh_properties
        }
        
        # Verify workflow consistency
        assert stellar_evolution['initial_mass'] == primary_mass
        assert orbital_mechanics['orbital_period'] > 0
        
        # Check evolutionary consistency
        ms_lifetime = stellar_evolution['main_sequence_lifetime']
        final_phase = stellar_evolution['final_phase']
        
        # 2 solar mass star should evolve off main sequence in ~5 Gyr
        assert ms_lifetime < 5e9
        
        # Orbital period should be reasonable for separation
        period_years = orbital_mechanics['orbital_period'] / (365.25 * 24 * 3600)
        assert 1 < period_years < 10  # Reasonable for given separation
    
    def test_cosmological_survey_simulation(self, astrophysics_workflow):
        """Test cosmological survey simulation workflow."""
        agent = astrophysics_workflow['agent']
        
        # Simulate observing supernovae at different redshifts
        redshifts = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2]
        survey_results = {}
        
        for z in redshifts:
            # Calculate cosmological distances
            cosmo_distances = agent.calculate_cosmological_distances(z)
            
            # Simulate supernova observation
            # Apparent magnitude = absolute magnitude + distance modulus
            absolute_magnitude = -19.3  # Type Ia standard candle
            apparent_magnitude = absolute_magnitude + cosmo_distances['distance_modulus']
            
            supernova_lc = agent.simulate_supernova_light_curve('Ia', apparent_magnitude)
            
            survey_results[z] = {
                'cosmology': cosmo_distances,
                'supernova': supernova_lc,
                'apparent_magnitude': apparent_magnitude
            }
        
        # Store survey results
        astrophysics_workflow['observations']['supernova_survey'] = survey_results
        
        # Analyze Hubble diagram
        redshifts_array = np.array(redshifts)
        distance_moduli = np.array([survey_results[z]['cosmology']['distance_modulus'] 
                                  for z in redshifts])
        
        # Check that distance modulus increases with redshift
        assert all(distance_moduli[i] <= distance_moduli[i+1] for i in range(len(distance_moduli)-1))
        
        # Check reasonable distance modulus range
        assert all(30 < dm < 50 for dm in distance_moduli)
        
        # Verify light curve consistency
        for z in redshifts:
            lc = survey_results[z]['supernova']
            assert lc['supernova_type'] == 'Ia'
            assert lc['peak_magnitude'] == survey_results[z]['apparent_magnitude']
    
    def test_galaxy_formation_simulation(self, astrophysics_workflow):
        """Test galaxy formation simulation workflow."""
        agent = astrophysics_workflow['agent']
        
        # Step 1: Dark matter halo simulation
        halo_mass = 1e12  # Solar masses
        galaxy_rotation = agent.simulate_galaxy_rotation_curve(halo_mass, dark_matter_fraction=0.85)
        
        # Step 2: N-body simulation of galaxy merger
        nbody_result = agent.simulate_nbody_system(n_bodies=100, box_size=50.0, n_steps=1000)
        
        # Step 3: CMB constraints
        cmb_result = agent.simulate_cosmic_microwave_background(l_max=2000)
        
        # Store galaxy formation results
        astrophysics_workflow['simulations']['galaxy_formation'] = {
            'rotation_curve': galaxy_rotation,
            'nbody_merger': nbody_result,
            'cmb_constraints': cmb_result
        }
        
        # Verify simulation consistency
        # Rotation curve should show dark matter effects
        assert galaxy_rotation['flat_rotation_evidence'] is True
        assert galaxy_rotation['dark_matter_fraction'] > 0.8
        
        # N-body simulation should conserve energy reasonably
        assert nbody_result['energy_conservation'] < 0.2
        
        # CMB should have acoustic peaks
        acoustic_peaks = cmb_result['peak_positions']
        assert len(acoustic_peaks) >= 3
        assert 200 < acoustic_peaks[0] < 250  # First acoustic peak
    
    @pytest.mark.integration
    def test_multi_messenger_astronomy(self, astrophysics_workflow):
        """Test multi-messenger astronomy workflow."""
        agent = astrophysics_workflow['agent']
        
        # Scenario: Binary black hole merger
        bh1_mass = 30.0  # Solar masses
        bh2_mass = 25.0  # Solar masses
        
        # Step 1: Pre-merger binary evolution
        separation = 1e9  # meters (very close)
        orbital_mechanics = agent.calculate_orbital_mechanics(
            bh1_mass * 1.989e30, bh2_mass * 1.989e30, separation
        )
        
        # Step 2: Individual black hole properties
        bh1_properties = agent.calculate_black_hole_properties(bh1_mass)
        bh2_properties = agent.calculate_black_hole_properties(bh2_mass)
        
        # Step 3: Merger remnant
        final_mass = bh1_mass + bh2_mass - 3.0  # Energy radiated in gravitational waves
        final_bh = agent.calculate_black_hole_properties(final_mass)
        
        # Step 4: Electromagnetic counterpart (if any)
        # For BBH merger, typically no EM counterpart, but test the capability
        counterpart_redshift = 0.1
        em_distances = agent.calculate_cosmological_distances(counterpart_redshift)
        
        # Store multi-messenger results
        astrophysics_workflow['observations']['multi_messenger'] = {
            'pre_merger_orbit': orbital_mechanics,
            'component_black_holes': [bh1_properties, bh2_properties],
            'final_black_hole': final_bh,
            'electromagnetic': em_distances,
            'gravitational_wave_strain': 1e-21  # Mock GW strain
        }
        
        # Verify multi-messenger consistency
        # Final BH should be less massive than sum (energy loss)
        assert final_bh['mass'] < bh1_mass + bh2_mass
        
        # Orbital frequency should be extremely high for close separation
        orbital_period = orbital_mechanics['orbital_period']
        orbital_frequency = 1 / orbital_period  # Hz
        assert orbital_frequency > 10  # Should be audible frequency for LIGO
        
        # Individual BH properties should be consistent
        assert bh1_properties['is_stellar_mass'] is True
        assert bh2_properties['is_stellar_mass'] is True
        assert final_bh['is_stellar_mass'] is True
    
    @pytest.mark.asyncio
    async def test_async_astrophysics_simulation(self, async_physics_simulator):
        """Test asynchronous astrophysics simulation."""
        # Start long N-body simulation
        simulation_task = asyncio.create_task(
            async_physics_simulator.run_simulation(duration=2.0, dt=0.001)
        )
        
        # Verify simulation is running
        assert async_physics_simulator.running is True
        
        # Wait for completion
        result = await simulation_task
        
        assert result['status'] == 'completed'
        assert result['final_time'] == 2.0
        assert async_physics_simulator.running is False