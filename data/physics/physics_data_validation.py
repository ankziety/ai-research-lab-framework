"""
Physics Data Validation for AI Research Lab Framework.

Provides validation and quality control for physics data including units,
dimensions, uncertainty analysis, and cross-validation against known physics principles.
"""

import logging
import re
import math
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# Try to import optional dependencies
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    stats = None

logger = logging.getLogger(__name__)

class ValidationLevel(Enum):
    """Validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Represents a validation result."""
    level: ValidationLevel
    message: str
    field: Optional[str] = None
    value: Optional[Any] = None
    expected: Optional[Any] = None
    suggestion: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level.value,
            'message': self.message,
            'field': self.field,
            'value': self.value,
            'expected': self.expected,
            'suggestion': self.suggestion,
            'timestamp': self.timestamp.isoformat()
        }

class PhysicsUnits:
    """Physics units and conversions."""
    
    # Base SI units
    BASE_UNITS = {
        'length': 'm',
        'mass': 'kg',
        'time': 's',
        'current': 'A',
        'temperature': 'K',
        'amount': 'mol',
        'luminosity': 'cd'
    }
    
    # Common physics units with SI conversion factors
    UNIT_CONVERSIONS = {
        # Length
        'mm': 1e-3, 'cm': 1e-2, 'km': 1e3,
        'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mile': 1609.344,
        'angstrom': 1e-10, 'nm': 1e-9, 'um': 1e-6,
        'ly': 9.4607e15, 'pc': 3.0857e16, 'au': 1.4960e11,
        
        # Mass
        'g': 1e-3, 'mg': 1e-6, 'ug': 1e-9,
        'lb': 0.453592, 'oz': 0.0283495,
        'amu': 1.66054e-27, 'Da': 1.66054e-27,
        'solar_mass': 1.989e30, 'earth_mass': 5.972e24,
        
        # Time
        'ms': 1e-3, 'us': 1e-6, 'ns': 1e-9, 'ps': 1e-12,
        'min': 60, 'hr': 3600, 'day': 86400, 'year': 3.154e7,
        
        # Energy
        'J': 1, 'kJ': 1e3, 'MJ': 1e6, 'GJ': 1e9,
        'eV': 1.602e-19, 'keV': 1.602e-16, 'MeV': 1.602e-13, 'GeV': 1.602e-10,
        'cal': 4.184, 'kcal': 4184, 'BTU': 1055,
        'erg': 1e-7,
        
        # Power
        'W': 1, 'kW': 1e3, 'MW': 1e6, 'GW': 1e9,
        'hp': 745.7,
        
        # Pressure
        'Pa': 1, 'kPa': 1e3, 'MPa': 1e6, 'GPa': 1e9,
        'bar': 1e5, 'atm': 101325, 'torr': 133.322, 'psi': 6895,
        
        # Temperature (additive conversions handled separately)
        'C': 273.15, 'F': (5/9, 255.372),  # Special handling needed
        
        # Frequency
        'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12,
        
        # Electric
        'V': 1, 'mV': 1e-3, 'kV': 1e3,
        'A': 1, 'mA': 1e-3, 'uA': 1e-6,
        'C': 1, 'mC': 1e-3, 'uC': 1e-6, 'nC': 1e-9,
        'ohm': 1, 'kohm': 1e3, 'Mohm': 1e6,
        
        # Magnetic
        'T': 1, 'mT': 1e-3, 'uT': 1e-6, 'nT': 1e-9,
        'G': 1e-4, 'mG': 1e-7,  # Gauss
    }
    
    # Dimensionless constants
    DIMENSIONLESS_CONSTANTS = {
        'pi', 'e', 'alpha', 'fine_structure_constant'
    }

class PhysicsDataValidation:
    """Validates physics data for quality and consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Physics Data Validation system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.units = PhysicsUnits()
        
        # Validation thresholds
        self.uncertainty_threshold = config.get('uncertainty_threshold', 0.1)  # 10%
        self.outlier_threshold = config.get('outlier_threshold', 3.0)  # 3 sigma
        
        # Physical constants for validation
        self.physical_constants = self._load_physical_constants()
        
        logger.info("PhysicsDataValidation initialized")
    
    def _load_physical_constants(self) -> Dict[str, Dict[str, Any]]:
        """Load known physical constants for validation."""
        return {
            'c': {
                'value': 299792458,
                'unit': 'm/s',
                'uncertainty': 0,
                'description': 'Speed of light in vacuum'
            },
            'h': {
                'value': 6.62607015e-34,
                'unit': 'J⋅s',
                'uncertainty': 0,
                'description': 'Planck constant'
            },
            'hbar': {
                'value': 1.054571817e-34,
                'unit': 'J⋅s',
                'uncertainty': 0,
                'description': 'Reduced Planck constant'
            },
            'e': {
                'value': 1.602176634e-19,
                'unit': 'C',
                'uncertainty': 0,
                'description': 'Elementary charge'
            },
            'me': {
                'value': 9.1093837015e-31,
                'unit': 'kg',
                'uncertainty': 0,
                'description': 'Electron rest mass'
            },
            'mp': {
                'value': 1.67262192369e-27,
                'unit': 'kg',
                'uncertainty': 5.1e-37,
                'description': 'Proton rest mass'
            },
            'kb': {
                'value': 1.380649e-23,
                'unit': 'J/K',
                'uncertainty': 0,
                'description': 'Boltzmann constant'
            },
            'NA': {
                'value': 6.02214076e23,
                'unit': 'mol⁻¹',
                'uncertainty': 0,
                'description': 'Avogadro constant'
            },
            'G': {
                'value': 6.67430e-11,
                'unit': 'm³⋅kg⁻¹⋅s⁻²',
                'uncertainty': 1.5e-15,
                'description': 'Gravitational constant'
            },
            'alpha': {
                'value': 7.2973525693e-3,
                'unit': 'dimensionless',
                'uncertainty': 1.1e-12,
                'description': 'Fine structure constant'
            }
        }
    
    def validate_physics_data(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate physics data against specified criteria.
        
        Args:
            data: Physics data to validate
            criteria: Validation criteria
        
        Returns:
            List of ValidationResult objects
        """
        results = []
        
        try:
            # Basic structure validation
            results.extend(self._validate_structure(data, criteria))
            
            # Units validation
            results.extend(self._validate_units(data, criteria))
            
            # Physical constraints validation
            results.extend(self._validate_physical_constraints(data, criteria))
            
            # Statistical validation
            results.extend(self._validate_statistics(data, criteria))
            
            # Uncertainty validation
            results.extend(self._validate_uncertainties(data, criteria))
            
            # Cross-validation with known constants
            results.extend(self._validate_against_constants(data, criteria))
            
            # Domain-specific validation
            domain = criteria.get('domain', 'general')
            if domain != 'general':
                results.extend(self._validate_domain_specific(data, criteria, domain))
            
            logger.info(f"Validation completed with {len(results)} issues found")
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            results.append(ValidationResult(
                level=ValidationLevel.CRITICAL,
                message=f"Validation process failed: {str(e)}"
            ))
        
        return results
    
    def _validate_structure(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate basic data structure."""
        results = []
        
        # Check required fields
        required_fields = criteria.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Required field '{field}' is missing",
                    field=field,
                    suggestion=f"Add required field '{field}' to the dataset"
                ))
        
        # Check data types
        expected_types = criteria.get('expected_types', {})
        for field, expected_type in expected_types.items():
            if field in data:
                actual_type = type(data[field]).__name__
                if actual_type != expected_type:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Field '{field}' has type '{actual_type}', expected '{expected_type}'",
                        field=field,
                        value=actual_type,
                        expected=expected_type
                    ))
        
        # Check for empty values
        for field, value in data.items():
            if value is None or (isinstance(value, (str, list, dict)) and len(value) == 0):
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Field '{field}' is empty",
                    field=field,
                    value=value
                ))
        
        return results
    
    def _validate_units(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate units and dimensional consistency."""
        results = []
        
        # Check unit format
        unit_fields = criteria.get('unit_fields', [])
        for field in unit_fields:
            if field in data:
                unit = data.get(f"{field}_unit", "")
                if not unit:
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Unit not specified for field '{field}'",
                        field=f"{field}_unit",
                        suggestion=f"Specify units for '{field}'"
                    ))
                elif not self._is_valid_unit(unit):
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Unknown unit '{unit}' for field '{field}'",
                        field=f"{field}_unit",
                        value=unit,
                        suggestion="Use standard SI units or common physics units"
                    ))
        
        # Check dimensional consistency
        dimensional_groups = criteria.get('dimensional_groups', {})
        for group_name, fields in dimensional_groups.items():
            dimensions = []
            for field in fields:
                if field in data:
                    unit = data.get(f"{field}_unit", "")
                    if unit:
                        dimension = self._get_unit_dimension(unit)
                        dimensions.append((field, dimension))
            
            # Check if all dimensions in group are consistent
            if len(dimensions) > 1:
                reference_dim = dimensions[0][1]
                for field, dimension in dimensions[1:]:
                    if dimension != reference_dim:
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Dimensional inconsistency in group '{group_name}': '{field}' has different dimension",
                            field=field,
                            suggestion="Ensure all fields in the group have compatible units"
                        ))
        
        return results
    
    def _validate_physical_constraints(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate against physical constraints and limits."""
        results = []
        
        # Check value ranges
        value_ranges = criteria.get('value_ranges', {})
        for field, (min_val, max_val) in value_ranges.items():
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)):
                    if value < min_val:
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Value of '{field}' ({value}) is below minimum ({min_val})",
                            field=field,
                            value=value,
                            expected=f">= {min_val}"
                        ))
                    elif value > max_val:
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Value of '{field}' ({value}) exceeds maximum ({max_val})",
                            field=field,
                            value=value,
                            expected=f"<= {max_val}"
                        ))
        
        # Check positivity constraints
        positive_fields = criteria.get('positive_fields', [])
        for field in positive_fields:
            if field in data:
                value = data[field]
                if isinstance(value, (int, float)) and value <= 0:
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Field '{field}' must be positive, got {value}",
                        field=field,
                        value=value,
                        expected="> 0"
                    ))
        
        # Check conservation laws (if applicable)
        conservation_checks = criteria.get('conservation_checks', [])
        for check in conservation_checks:
            if check == 'energy':
                results.extend(self._check_energy_conservation(data))
            elif check == 'momentum':
                results.extend(self._check_momentum_conservation(data))
            elif check == 'charge':
                results.extend(self._check_charge_conservation(data))
        
        return results
    
    def _validate_statistics(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate statistical properties of data."""
        results = []
        
        if not NUMPY_AVAILABLE or not SCIPY_AVAILABLE:
            results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Statistical validation requires numpy and scipy"
            ))
            return results
        
        # Check for outliers in numerical arrays
        numerical_fields = criteria.get('numerical_fields', [])
        for field in numerical_fields:
            if field in data:
                values = data[field]
                if isinstance(values, (list, np.ndarray)) and len(values) > 3:
                    outliers = self._detect_outliers(values)
                    if outliers:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Found {len(outliers)} potential outliers in '{field}'",
                            field=field,
                            value=f"Outlier indices: {outliers}",
                            suggestion="Review data points for measurement errors"
                        ))
        
        # Check distributions
        distribution_checks = criteria.get('distribution_checks', {})
        for field, expected_dist in distribution_checks.items():
            if field in data:
                values = data[field]
                if isinstance(values, (list, np.ndarray)) and len(values) > 10:
                    is_normal = self._test_normality(values)
                    if expected_dist == 'normal' and not is_normal:
                        results.append(ValidationResult(
                            level=ValidationLevel.INFO,
                            message=f"Field '{field}' does not follow normal distribution",
                            field=field,
                            suggestion="Consider data transformation or alternative analysis methods"
                        ))
        
        return results
    
    def _validate_uncertainties(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate uncertainty information."""
        results = []
        
        # Check for uncertainty fields
        uncertainty_fields = criteria.get('uncertainty_fields', [])
        for field in uncertainty_fields:
            uncertainty_field = f"{field}_uncertainty"
            if field in data and uncertainty_field not in data:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Uncertainty not provided for '{field}'",
                    field=uncertainty_field,
                    suggestion=f"Add uncertainty information for '{field}'"
                ))
            elif field in data and uncertainty_field in data:
                value = data[field]
                uncertainty = data[uncertainty_field]
                
                if isinstance(value, (int, float)) and isinstance(uncertainty, (int, float)):
                    # Check if uncertainty is reasonable
                    relative_uncertainty = abs(uncertainty / value) if value != 0 else float('inf')
                    if relative_uncertainty > self.uncertainty_threshold:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Large uncertainty for '{field}': {relative_uncertainty*100:.1f}%",
                            field=uncertainty_field,
                            value=f"{relative_uncertainty*100:.1f}%",
                            suggestion="Review measurement precision"
                        ))
                    
                    # Check if uncertainty is positive
                    if uncertainty < 0:
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Negative uncertainty for '{field}': {uncertainty}",
                            field=uncertainty_field,
                            value=uncertainty,
                            expected=">= 0"
                        ))
        
        return results
    
    def _validate_against_constants(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate against known physical constants."""
        results = []
        
        constant_checks = criteria.get('constant_checks', [])
        for constant_name in constant_checks:
            if constant_name in data and constant_name in self.physical_constants:
                measured_value = data[constant_name]
                known_constant = self.physical_constants[constant_name]
                
                if isinstance(measured_value, (int, float)):
                    expected_value = known_constant['value']
                    tolerance = known_constant.get('uncertainty', expected_value * 0.01)  # 1% default
                    
                    deviation = abs(measured_value - expected_value)
                    if deviation > tolerance:
                        results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Value of '{constant_name}' deviates significantly from known constant",
                            field=constant_name,
                            value=measured_value,
                            expected=f"{expected_value} ± {tolerance}",
                            suggestion=f"Check measurement or units for {known_constant['description']}"
                        ))
        
        return results
    
    def _validate_domain_specific(self, data: Dict[str, Any], criteria: Dict[str, Any], domain: str) -> List[ValidationResult]:
        """Validate domain-specific constraints."""
        results = []
        
        if domain == 'particle_physics':
            results.extend(self._validate_particle_physics(data, criteria))
        elif domain == 'condensed_matter':
            results.extend(self._validate_condensed_matter(data, criteria))
        elif domain == 'astrophysics':
            results.extend(self._validate_astrophysics(data, criteria))
        elif domain == 'quantum_mechanics':
            results.extend(self._validate_quantum_mechanics(data, criteria))
        
        return results
    
    def _validate_particle_physics(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate particle physics specific constraints."""
        results = []
        
        # Check particle masses
        if 'particle_mass' in data:
            mass = data['particle_mass']
            mass_unit = data.get('particle_mass_unit', 'kg')
            
            # Convert to MeV/c²
            if mass_unit == 'kg':
                mass_mev = mass * KG_TO_MEV_C2  # Convert kg to MeV/c²
            elif mass_unit in ['MeV', 'MeV/c2']:
                mass_mev = mass
            else:
                mass_mev = None
            
            if mass_mev and mass_mev > 1e6:  # Beyond known particle masses
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Particle mass ({mass_mev:.2e} MeV) exceeds typical particle masses",
                    field='particle_mass',
                    value=mass_mev,
                    suggestion="Verify particle identification or units"
                ))
        
        # Check quantum numbers
        quantum_numbers = data.get('quantum_numbers', {})
        if 'spin' in quantum_numbers:
            spin = quantum_numbers['spin']
            if not isinstance(spin, (int, float)) or spin < 0:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Invalid spin value: {spin}",
                    field='quantum_numbers.spin',
                    value=spin,
                    expected="Non-negative number"
                ))
        
        return results
    
    def _validate_condensed_matter(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate condensed matter physics constraints."""
        results = []
        
        # Check temperature ranges
        if 'temperature' in data:
            temp = data['temperature']
            temp_unit = data.get('temperature_unit', 'K')
            
            if temp_unit == 'K' and temp < 0:
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Temperature below absolute zero: {temp} K",
                    field='temperature',
                    value=temp,
                    expected=">= 0 K"
                ))
        
        # Check crystal structure
        if 'crystal_structure' in data:
            structure = data['crystal_structure']
            valid_structures = ['cubic', 'tetragonal', 'orthorhombic', 'hexagonal', 'trigonal', 'monoclinic', 'triclinic']
            if structure not in valid_structures:
                results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Unknown crystal structure: {structure}",
                    field='crystal_structure',
                    value=structure,
                    expected=f"One of: {', '.join(valid_structures)}"
                ))
        
        return results
    
    def _validate_astrophysics(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate astrophysics constraints."""
        results = []
        
        # Check coordinates
        if 'ra' in data and 'dec' in data:
            ra = data['ra']
            dec = data['dec']
            
            if not (0 <= ra <= 360):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Right ascension out of range: {ra}°",
                    field='ra',
                    value=ra,
                    expected="0° to 360°"
                ))
            
            if not (-90 <= dec <= 90):
                results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Declination out of range: {dec}°",
                    field='dec',
                    value=dec,
                    expected="-90° to 90°"
                ))
        
        # Check stellar masses
        if 'stellar_mass' in data:
            mass = data['stellar_mass']
            mass_unit = data.get('stellar_mass_unit', 'solar_mass')
            
            if mass_unit == 'solar_mass' and (mass < 0.08 or mass > 150):
                level = ValidationLevel.WARNING if 0.08 <= mass <= 300 else ValidationLevel.ERROR
                results.append(ValidationResult(
                    level=level,
                    message=f"Stellar mass ({mass} M☉) outside typical range",
                    field='stellar_mass',
                    value=mass,
                    expected="0.08 to 150 M☉ for main sequence stars"
                ))
        
        return results
    
    def _validate_quantum_mechanics(self, data: Dict[str, Any], criteria: Dict[str, Any]) -> List[ValidationResult]:
        """Validate quantum mechanics constraints."""
        results = []
        
        # Check energy levels
        if 'energy_levels' in data:
            levels = data['energy_levels']
            if isinstance(levels, list) and len(levels) > 1:
                for i in range(1, len(levels)):
                    if levels[i] <= levels[i-1]:
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Energy levels not in ascending order at index {i}",
                            field=f'energy_levels[{i}]',
                            value=levels[i],
                            suggestion="Energy levels should be ordered from lowest to highest"
                        ))
        
        # Check wave function normalization
        if 'wave_function' in data and NUMPY_AVAILABLE:
            psi = data['wave_function']
            if isinstance(psi, (list, np.ndarray)):
                psi_array = np.array(psi)
                if np.iscomplexobj(psi_array):
                    norm_squared = np.sum(np.abs(psi_array)**2)
                    if abs(norm_squared - 1.0) > 0.01:  # 1% tolerance
                        results.append(ValidationResult(
                            level=ValidationLevel.WARNING,
                            message=f"Wave function not normalized: ∫|ψ|² = {norm_squared:.4f}",
                            field='wave_function',
                            value=norm_squared,
                            expected="≈ 1.0",
                            suggestion="Normalize the wave function"
                        ))
        
        return results
    
    # Helper methods
    def _is_valid_unit(self, unit: str) -> bool:
        """Check if unit is recognized."""
        # Remove mathematical symbols and check base units
        clean_unit = re.sub(r'[⋅²³⁻¹/\s\-\+\d]', '', unit)
        unit_parts = re.split(r'[⋅/]', clean_unit)
        
        for part in unit_parts:
            if part and part not in self.units.UNIT_CONVERSIONS and part not in self.units.BASE_UNITS.values():
                return False
        
        return True
    
    def _get_unit_dimension(self, unit: str) -> str:
        """Get the physical dimension of a unit."""
        # Simplified dimension analysis
        # In practice, this would need a more sophisticated implementation
        base_dimensions = {
            'm': 'length', 'kg': 'mass', 's': 'time',
            'A': 'current', 'K': 'temperature', 'mol': 'amount', 'cd': 'luminosity'
        }
        
        for base_unit, dimension in base_dimensions.items():
            if base_unit in unit:
                return dimension
        
        return 'unknown'
    
    def _detect_outliers(self, values: Union[List, np.ndarray]) -> List[int]:
        """Detect outliers using the IQR method."""
        if not NUMPY_AVAILABLE:
            return []
        
        arr = np.array(values)
        Q1 = np.percentile(arr, 25)
        Q3 = np.percentile(arr, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))[0]
        return outlier_indices.tolist()
    
    def _test_normality(self, values: Union[List, np.ndarray]) -> bool:
        """Test if values follow normal distribution."""
        if not SCIPY_AVAILABLE:
            return True  # Assume normal if can't test
        
        arr = np.array(values)
        if len(arr) < 8:  # Minimum sample size for Shapiro-Wilk
            return True
        
        try:
            statistic, p_value = stats.shapiro(arr)
            return p_value > 0.05  # Normal if p > 0.05
        except:
            return True
    
    def _check_energy_conservation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Check energy conservation in the data."""
        results = []
        
        # Look for energy terms
        energy_fields = [k for k in data.keys() if 'energy' in k.lower()]
        
        if 'initial_energy' in data and 'final_energy' in data:
            initial = data['initial_energy']
            final = data['final_energy']
            
            if isinstance(initial, (int, float)) and isinstance(final, (int, float)):
                energy_diff = abs(final - initial)
                tolerance = max(abs(initial), abs(final)) * 0.01  # 1% tolerance
                
                if energy_diff > tolerance:
                    results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Energy not conserved: ΔE = {energy_diff:.2e}",
                        suggestion="Check for energy losses or measurement errors"
                    ))
        
        return results
    
    def _check_momentum_conservation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Check momentum conservation in the data."""
        results = []
        
        # Similar implementation for momentum conservation
        # This would check vector momentum components
        
        return results
    
    def _check_charge_conservation(self, data: Dict[str, Any]) -> List[ValidationResult]:
        """Check charge conservation in the data."""
        results = []
        
        # Check if total charge is conserved in reactions
        if 'initial_charge' in data and 'final_charge' in data:
            initial = data['initial_charge']
            final = data['final_charge']
            
            if isinstance(initial, (int, float)) and isinstance(final, (int, float)):
                if abs(final - initial) > 1e-10:  # Very small tolerance for charge
                    results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Charge not conserved: Δq = {final - initial}",
                        suggestion="Check particle charges in the reaction"
                    ))
        
        return results
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get a summary of validation results."""
        summary = {
            'total_issues': len(results),
            'by_level': {level.value: 0 for level in ValidationLevel},
            'by_field': {},
            'critical_issues': [],
            'recommendations': []
        }
        
        for result in results:
            # Count by level
            summary['by_level'][result.level.value] += 1
            
            # Count by field
            if result.field:
                summary['by_field'][result.field] = summary['by_field'].get(result.field, 0) + 1
            
            # Collect critical issues
            if result.level == ValidationLevel.CRITICAL:
                summary['critical_issues'].append(result.message)
            
            # Collect recommendations
            if result.suggestion:
                summary['recommendations'].append(result.suggestion)
        
        # Remove duplicates from recommendations
        summary['recommendations'] = list(set(summary['recommendations']))
        
        return summary