# HiPoSa: Hierarchical Poisson Sampling

A Python library for generating hierarchical Poisson disk sampling patterns with support for symmetry operations, tiling, and multi-dimensional spaces.

## Overview

HiPoSa (Hierarchical Poisson Sampling) is a sophisticated library for generating well-distributed point patterns using Poisson disk sampling with advanced features including:

- **Multi-dimensional support** (2D, 3D, 4D, etc.)
- **Hierarchical sampling** with multiple spacing levels
- **Symmetry operations** (rotational, translational, custom)
- **Periodic boundary conditions** for seamless tiling
- **Point selection** based on interpolated data and thresholds
- **Parallel processing** for large-scale tiling operations

The library is designed for scientific applications requiring well-distributed sampling patterns, such as experimental design, numerical integration, and spatial analysis.

## Architecture

HiPoSa consists of three main components that work together to create sophisticated sampling patterns:

1. **PoissonDiskSamplerWithExisting**: Core sampling engine with symmetry support
2. **PoissonTiler**: Hierarchical tiling system for large-scale patterns
3. **PointSelector**: Intelligent point selection based on data-driven criteria

The system uses KDTree-based nearest neighbor searches for efficient distance calculations and supports both periodic and non-periodic boundary conditions.

## Key Components

### PoissonDiskSamplerWithExisting
**Purpose**: Core Poisson disk sampling engine with support for existing points and symmetry operations
**Key Functions**:
- `sample()`: Generate new points while respecting minimum distance constraints
- `generate_points_around()`: Create candidate points around existing samples
- `apply_symmetry()`: Apply symmetry operations to generate symmetric point sets
- `find_invariant_points()`: Locate points that are invariant under symmetry operations

**Input/Output**: Takes domain boundaries, minimum distance, and optional existing points; returns arrays of points and labels
**Dependencies**: NumPy, SciPy (KDTree, optimize)

### PoissonTiler
**Purpose**: Creates hierarchical sampling patterns that can be tiled across large areas
**Key Functions**:
- `get_points_in_region()`: Extract points from a specific region with parallel processing
- `_generate_base_tile()`: Create the base tile with hierarchical spacing levels
- `_process_tile()`: Process individual tiles for parallel computation

**Input/Output**: Takes tile size and spacing levels; returns hierarchical point patterns
**Dependencies**: PoissonDiskSamplerWithExisting, multiprocessing

### PointSelector
**Purpose**: Selects points based on interpolated data and threshold criteria
**Key Functions**:
- `run()`: Execute the complete point selection algorithm
- `compute_threshold()`: Calculate thresholds based on working and calibration sets
- `select_points_at_level()`: Select points at specific hierarchical levels
- `plot_results()`: Visualize selection results with matplotlib

**Input/Output**: Takes point coordinates, levels, and evaluation function; returns selected point indices
**Dependencies**: NumPy, SciPy (interpolation), matplotlib, scikit-learn

### Configuration System
**Purpose**: Centralized configuration management for all components
**Key Classes**:
- `SamplingConfig`: Parameters for Poisson disk sampling
- `TilingConfig`: Settings for hierarchical tiling
- `PointSelectorConfig`: Threshold and selection parameters

## Usage Examples

### Basic Usage
```python
# CODE_EXAMPLE_START
# Simple Poisson disk sampling in 2D
import numpy as np
from hiposa import PoissonDiskSamplerWithExisting

# Define domain and minimum distance
domain = [(0, 10), (0, 10)]
r = 0.5

# Create sampler and generate points
sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
points, labels = sampler.sample()

print(f"Generated {len(points)} points")
# CODE_EXAMPLE_STOP
```

### Advanced Usage with Symmetry
```python
# CODE_EXAMPLE_START
# Poisson disk sampling with rotational symmetry
import numpy as np
from hiposa import PoissonDiskSamplerWithExisting
from hiposa.utils import create_symmetry_operators

# Define domain and parameters
domain = [(0, 10), (0, 10)]
r = 0.5

# Create 90-degree rotation symmetry
rotation_angles = [np.pi/2, np.pi, 3*np.pi/2]
symmetry_operators = create_symmetry_operators(rotation_angles=rotation_angles)

# Create sampler with symmetry
sampler = PoissonDiskSamplerWithExisting(
    domain=domain, 
    r=r, 
    symmetry_operators=symmetry_operators
)

points, labels = sampler.sample()
print(f"Generated {len(points)} points with 4-fold rotational symmetry")
# CODE_EXAMPLE_STOP
```

### Hierarchical Tiling
```python
# CODE_EXAMPLE_START
# Create hierarchical tiling pattern
import numpy as np
from hiposa import PoissonTiler

# Define tile parameters
tile_size = 10.0
spacings = [2.0, 1.0, 0.5]  # Hierarchical spacing levels

# Create tiler
tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)

# Extract points from a specific region
region = [(0, 30), (0, 20)]
points, levels = tiler.get_points_in_region(region, n_processes=4)

print(f"Generated {len(points)} points across {len(spacings)} levels")
print(f"Level distribution: {np.bincount(levels)}")
# CODE_EXAMPLE_STOP
```

### Point Selection with Data-Driven Criteria
```python
# CODE_EXAMPLE_START
# Select points based on function evaluation
import numpy as np
from hiposa import PointSelector

# Define evaluation function
def test_function(point):
    x, y = point
    return np.sin(x) * np.cos(y) + 0.1 * np.random.randn()

# Create test data
n_points = 100
xy = np.random.rand(n_points, 2) * 10
levels = np.random.randint(0, 3, n_points)
scales = [1.0, 0.5, 0.25]

# Create grid for interpolation
grid_x = np.linspace(0, 10, 50)
grid_y = np.linspace(0, 10, 50)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# Create selector and run
selector = PointSelector(
    xy=xy,
    levels=levels,
    scales=scales,
    f_function=test_function,
    grid_x=grid_x,
    grid_y=grid_y,
    tau=75.0  # 75th percentile threshold
)

selected_points = selector.run(max_level=3)
print(f"Selected {np.sum(selected_points)} points based on function evaluation")
# CODE_EXAMPLE_STOP
```

## API Reference

### Functions

#### PoissonDiskSamplerWithExisting
- `__init__(domain: List[Tuple[float, float]], r: float, existing_points: Optional[np.ndarray] = None, existing_labels: Optional[np.ndarray] = None, k: int = 60, symmetry_operators: Optional[List[Callable]] = None, wrap: bool = False)`: Initialize sampler with domain and parameters
- `sample(new_label: Optional[Union[int, str]] = None, return_new_only: bool = False) -> Tuple[np.ndarray, np.ndarray]`: Generate new points and return with labels
- `generate_points_around(point: np.ndarray) -> np.ndarray`: Generate candidate points around existing point
- `is_valid_point(point: np.ndarray) -> bool`: Check if point satisfies minimum distance constraint
- `apply_symmetry(point: np.ndarray) -> Optional[List[np.ndarray]]`: Apply symmetry operations to point
- `find_invariant_points(operator: Callable) -> List[np.ndarray]`: Find points invariant under symmetry operator

#### PoissonTiler
- `__init__(tile_size: float, spacings: List[float], dimensions: int = 2, min_tile_factor: float = 2.0)`: Initialize tiler with tile size and spacing levels
- `get_points_in_region(region: List[Tuple[float, float]], n_processes: Optional[int] = None, add_corners: bool = True) -> Tuple[np.ndarray, np.ndarray]`: Extract points from specified region
- `_generate_base_tile() -> None`: Generate the base tile with hierarchical sampling
- `_process_tile(args: Tuple) -> Tuple[np.ndarray, np.ndarray]`: Process individual tile (for parallel processing)

#### PointSelector
- `__init__(xy: npt.NDArray, levels: npt.NDArray, scales: npt.NDArray, f_function: Callable, grid_x: npt.NDArray, grid_y: npt.NDArray, f_gt: Optional[npt.NDArray] = None, tau: float = 75.0, sign: int = 1, eps: float = 0.0, start_level: int = 0, set_aside: float = 0.5, lower: Optional[float] = None, upper: Optional[float] = None, lower_tau_quantile: float = 1.0, n_splits_quantile: int = 10, border: int = 4, border_bias: float = 0.5, radius_scale: float = 4.0)`: Initialize selector with data and parameters
- `run(max_level: int = 19) -> npt.NDArray`: Execute point selection algorithm
- `compute_threshold(work_xy: npt.NDArray, work_f_values: npt.NDArray, cal_f_values: npt.NDArray) -> Tuple[float, npt.NDArray, float, float, float]`: Calculate threshold based on working and calibration sets
- `select_points_at_level(level: int, threshold: float, grid_values: npt.NDArray) -> npt.NDArray`: Select points at specific level
- `plot_results(these_xy: npt.NDArray, new_ones: npt.NDArray, threshold: float, surface: npt.NDArray, mask: npt.NDArray, title: Optional[str] = None, level: Optional[int] = None) -> None`: Visualize selection results

### Classes

#### SamplingConfig
Configuration for Poisson disk sampling parameters:
- `DEFAULT_K: int = 60`: Maximum attempts to generate new points
- `DEFAULT_K_SECOND_PASS: int = 80`: K value for second sampling pass
- `DEFAULT_EPSILON: float = 1e-10`: Numerical tolerance for orbit validation
- `KDTree_UPDATE_FREQUENCY: int = 10`: Frequency of KDTree updates
- `MINIMIZATION_XATOL: float = 1e-10`: Tolerance for optimization
- `MINIMIZATION_FATOL: float = 1e-10`: Function tolerance for optimization
- `INVARIANT_POINT_ATTEMPTS: int = 10`: Attempts to find invariant points

#### TilingConfig
Configuration for hierarchical tiling:
- `MIN_TILE_FACTOR: float = 2.0`: Minimum tile size factor relative to largest spacing
- `DEFAULT_N_PROCESSES: int = None`: Default number of processes for parallel processing
- `DEFAULT_ADD_CORNERS: bool = True`: Whether to add corner points by default

#### PointSelectorConfig
Configuration for point selection:
- `DEFAULT_TAU: float = 75.0`: Default percentile for thresholding
- `DEFAULT_SIGN: int = 1`: Direction of thresholding (1 for greater than, -1 for less than)
- `DEFAULT_EPS: float = 0.0`: Small adjustment to threshold
- `DEFAULT_START_LEVEL: int = 0`: Level to start selection process
- `DEFAULT_SET_ASIDE: float = 0.5`: Fraction of points to set aside for calibration
- `DEFAULT_LOWER_TAU_QUANTILE: float = 1.0`: Lower quantile for threshold adjustment
- `DEFAULT_N_SPLITS_QUANTILE: int = 10`: Number of splits for cross-validation
- `DEFAULT_BORDER: int = 4`: Border size for masking
- `DEFAULT_BORDER_BIAS: float = 0.5`: Bias towards sampling border regions
- `DEFAULT_RADIUS_SCALE: float = 4.0`: Multiplier for effective radius in border regions
- `DEFAULT_N_SPLITS: int = 5`: Number of splits for cross-validation

## Configuration

HiPoSa includes an optional centralized configuration system in `config.py` that provides default parameters for all components. The config system is **optional** - if not used, components fall back to their hardcoded defaults.

### Using the Config System

```python
# CODE_EXAMPLE_START
# Access default configuration
from hiposa.config import get_default_config, SAMPLING_CONFIG, POINT_SELECTOR_CONFIG

# Get all default parameters
config = get_default_config()

# Modify specific parameters
SAMPLING_CONFIG.DEFAULT_K = 80  # Increase sampling attempts
POINT_SELECTOR_CONFIG.DEFAULT_TAU = 80.0  # Change threshold percentile

# Components will now use these modified defaults
# CODE_EXAMPLE_STOP
```

### Config Integration

The config system is **gracefully integrated** - components will use config defaults if available, but fall back to hardcoded defaults if the config system is not available. This ensures backward compatibility and allows the library to work even if the config module is removed.

**Components with config integration:**
- `PointSelector`: Full integration with all parameters
- `PoissonDiskSamplerWithExisting`: Partial integration (k parameter)
- `PoissonTiler`: No integration (keeps simple defaults)

### Environment Variables
- No environment variables are currently required
- All configuration is handled through the config system

### Settings Files
- Configuration is centralized in `hiposa/config.py`
- Parameters can be modified at runtime through config objects

## Data Flow

### Input Formats
1. **Domain Definition**: List of (min, max) tuples for each dimension
2. **Point Arrays**: 2D NumPy arrays with shape (n_points, n_dimensions)
3. **Label Arrays**: 1D NumPy arrays or lists of labels
4. **Symmetry Operators**: List of callable functions that transform points

### Processing Steps
1. **Initialization**: Validate domain and parameters, setup KDTree if existing points provided
2. **Point Generation**: Generate candidate points around existing samples
3. **Symmetry Application**: Apply symmetry operations to create symmetric point sets
4. **Validation**: Check minimum distance constraints and domain boundaries
5. **Hierarchical Processing**: For tiling, process multiple spacing levels
6. **Point Selection**: For PointSelector, interpolate data and apply thresholds

### Output Formats
1. **Points**: 2D NumPy array with shape (n_points, n_dimensions)
2. **Labels**: 1D NumPy array or list of labels corresponding to points
3. **Selection Masks**: Boolean arrays indicating selected points
4. **Visualization**: Matplotlib plots showing sampling patterns and selection results

### Transformations Applied
- **Symmetry Operations**: Rotational, translational, and custom transformations
- **Periodic Wrapping**: For tiling applications, points wrap around domain boundaries
- **Interpolation**: Sparse data is interpolated to regular grids for analysis
- **Thresholding**: Data-driven point selection based on percentile thresholds

## Common Use Cases

### 1. Experimental Design
Generate well-distributed sampling points for experimental design:
```python
# CODE_EXAMPLE_START
# Create experimental design with symmetry
import numpy as np
from hiposa import PoissonDiskSamplerWithExisting
from hiposa.utils import create_symmetry_operators

# Define experimental domain
domain = [(0, 1), (0, 1)]  # Unit square
r = 0.1  # Minimum distance between experiments

# Add rotational symmetry for efficiency
symmetry_operators = create_symmetry_operators(rotation_angles=[np.pi/2])

# Generate experimental points
sampler = PoissonDiskSamplerWithExisting(
    domain=domain, 
    r=r, 
    symmetry_operators=symmetry_operators
)
experimental_points, labels = sampler.sample()

print(f"Generated {len(experimental_points)} experimental points")
# CODE_EXAMPLE_STOP
```

### 2. Large-Scale Spatial Sampling
Create hierarchical sampling patterns for large areas:
```python
# CODE_EXAMPLE_START
# Hierarchical sampling for large spatial domain
import numpy as np
from hiposa import PoissonTiler

# Define hierarchical spacing levels
tile_size = 100.0
spacings = [20.0, 10.0, 5.0, 2.5]  # Coarse to fine

# Create tiler for large area
tiler = PoissonTiler(tile_size=tile_size, spacings=spacings)

# Sample large region efficiently
large_region = [(0, 500), (0, 300)]
points, levels = tiler.get_points_in_region(
    large_region, 
    n_processes=8,  # Parallel processing
    add_corners=True
)

print(f"Sampled {len(points)} points across {len(spacings)} levels")
print(f"Point density: {len(points) / (500 * 300):.2f} points per unit area")
# CODE_EXAMPLE_STOP
```

### 3. Adaptive Sampling Based on Data
Select sampling points based on data-driven criteria:
```python
# CODE_EXAMPLE_START
# Adaptive sampling based on function behavior
import numpy as np
from hiposa import PointSelector

# Define function with varying complexity
def complex_function(point):
    x, y = point
    # High complexity near origin, low complexity elsewhere
    return np.exp(-(x**2 + y**2)) + 0.1 * np.random.randn()

# Create initial sampling grid
n_initial = 200
xy = np.random.rand(n_initial, 2) * 4 - 2  # [-2, 2] domain
levels = np.random.randint(0, 4, n_initial)
scales = [1.0, 0.5, 0.25, 0.125]

# Create interpolation grid
grid_x = np.linspace(-2, 2, 100)
grid_y = np.linspace(-2, 2, 100)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# Adaptive point selection
selector = PointSelector(
    xy=xy,
    levels=levels,
    scales=scales,
    f_function=complex_function,
    grid_x=grid_x,
    grid_y=grid_y,
    tau=80.0,  # 80th percentile threshold
    border_bias=0.7  # Bias towards complex regions
)

selected = selector.run(max_level=3)
print(f"Adaptively selected {np.sum(selected)} points")
# CODE_EXAMPLE_STOP
```

## Troubleshooting

### Common Issues

1. **Issue**: "Domain cannot be empty" error
   - **Cause**: Empty domain list passed to sampler
   - **Solution**: Ensure domain is a non-empty list of (min, max) tuples
   - **Prevention**: Validate domain before creating sampler
   ```python
   # CODE_EXAMPLE_START
   # Correct domain definition
   from hiposa.utils import validate_domain
   
   domain = [(0, 10), (0, 10)]  # 2D domain
   validate_domain(domain)  # Validates domain before use
   
   sampler = PoissonDiskSamplerWithExisting(domain=domain, r=0.5)
   # CODE_EXAMPLE_STOP
   ```

2. **Issue**: "Minimum distance r must be positive" error
   - **Cause**: Non-positive minimum distance parameter
   - **Solution**: Use positive value for minimum distance
   - **Code Example**: 
   ```python
   # CODE_EXAMPLE_START
   # Correct minimum distance
   r = 0.5  # Positive minimum distance
   sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
   # CODE_EXAMPLE_STOP
   ```

3. **Issue**: Poor sampling density or gaps
   - **Cause**: Minimum distance too large relative to domain size
   - **Solution**: Reduce minimum distance or increase domain size
   - **Prevention**: Use `estimate_optimal_spacing()` utility
   ```python
   # CODE_EXAMPLE_START
   # Estimate optimal spacing
   from hiposa.utils import estimate_optimal_spacing
   
   domain = [(0, 10), (0, 10)]
   target_points = 100
   optimal_r = estimate_optimal_spacing(domain, target_points)
   print(f"Optimal spacing: {optimal_r:.3f}")
   # CODE_EXAMPLE_STOP
   ```

4. **Issue**: Symmetry operations not working as expected
   - **Cause**: Symmetry operators not properly defined or incompatible with domain
   - **Solution**: Ensure symmetry operators preserve domain boundaries
   - **Prevention**: Use utility functions for common symmetries
   ```python
   # CODE_EXAMPLE_START
   # Proper symmetry definition
   from hiposa.utils import create_symmetry_operators
   
   # Use utility function for common symmetries
   rotation_angles = [np.pi/2, np.pi, 3*np.pi/2]
   symmetry_operators = create_symmetry_operators(rotation_angles=rotation_angles)
   
   sampler = PoissonDiskSamplerWithExisting(
       domain=domain, 
       r=r, 
       symmetry_operators=symmetry_operators
   )
   # CODE_EXAMPLE_STOP
   ```

5. **Issue**: Tiling produces gaps or overlaps
   - **Cause**: Tile size too small relative to spacing levels
   - **Solution**: Increase tile size or reduce spacing levels
   - **Prevention**: Use appropriate `min_tile_factor`
   ```python
   # CODE_EXAMPLE_START
   # Proper tiling setup
   tile_size = 10.0
   spacings = [2.0, 1.0, 0.5]
   min_tile_factor = 3.0  # Ensure tile size is 3x largest spacing
   
   tiler = PoissonTiler(
       tile_size=tile_size, 
       spacings=spacings, 
       min_tile_factor=min_tile_factor
   )
   # CODE_EXAMPLE_STOP
   ```

### Error Messages

- `ValueError: Domain cannot be empty`: Domain list is empty or None
- `ValueError: Minimum distance r must be positive`: Non-positive minimum distance
- `ValueError: Point dimension X does not match domain dimension Y`: Point dimensionality mismatch
- `RuntimeError: No valid points found`: Unable to generate valid points with current parameters
- `ImportError: No module named 'scipy'`: Missing SciPy dependency

### Performance Tips

- **KDTree Updates**: Reduce `KDTree_UPDATE_FREQUENCY` for better performance with large point sets
- **Parallel Processing**: Use `n_processes` parameter in `get_points_in_region()` for large tiling operations
- **Memory Management**: For very large domains, process regions incrementally
- **Symmetry Efficiency**: Use symmetry operations to reduce computational load
- **Grid Resolution**: Balance interpolation grid resolution with computational cost in PointSelector

## Dependencies

### Core Dependencies
- `numpy>=1.17.0`: Numerical computing and array operations
- `scipy>=1.3.0`: Scientific computing (KDTree, optimization, interpolation)
- `matplotlib>=3.0.0`: Visualization and plotting
- `scikit-learn>=0.24.0`: Machine learning utilities for PointSelector

### Optional Dependencies
- `jupyter>=1.0`: Jupyter notebook support for examples
- `ipywidgets>=7.0`: Interactive widgets for examples
- `seaborn>=0.11.0`: Enhanced plotting for examples
- `pytest>=6.0`: Testing framework
- `black>=22.0`: Code formatting
- `mypy>=0.950`: Type checking

## Testing

Test the functionality with provided test cases:

```python
# CODE_EXAMPLE_START
# Example test cases
import pytest
import numpy as np
from hiposa import PoissonDiskSamplerWithExisting
from hiposa.utils import check_minimum_distance

def test_basic_sampling():
    """Test basic Poisson disk sampling."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    sampler = PoissonDiskSamplerWithExisting(domain=domain, r=r)
    points, labels = sampler.sample()
    
    assert len(points) > 0, "Should generate points"
    assert check_minimum_distance(points, r), "Should maintain minimum distance"
    assert points.shape[1] == 2, "Should be 2D points"

def test_symmetry_sampling():
    """Test sampling with symmetry operations."""
    domain = [(0, 10), (0, 10)]
    r = 0.5
    
    # 90-degree rotation
    def rotate_90(point):
        x, y = point
        return np.array([-y, x])
    
    sampler = PoissonDiskSamplerWithExisting(
        domain=domain, 
        r=r, 
        symmetry_operators=[rotate_90]
    )
    points, labels = sampler.sample()
    
    assert len(points) > 0, "Should generate symmetric points"
# CODE_EXAMPLE_STOP
```

Run tests with:
```bash
pytest tests/ -v
```

## Integration Notes

### Upstream Components
- **Domain Definition**: Expects well-defined domain boundaries
- **Function Evaluation**: PointSelector expects callable functions for data evaluation
- **Grid Generation**: Requires regular grids for interpolation

### Downstream Components
- **Point Arrays**: Provides NumPy arrays compatible with most scientific libraries
- **Label Arrays**: Integer or string labels for point classification
- **Visualization**: Matplotlib-compatible plotting functions

### Shared Resources
- **Configuration**: Centralized config system accessible to all components
- **Logging**: Unified logging system for debugging and monitoring
- **Validation**: Common validation utilities for domain and point arrays

## Changelog/Version Notes

### Version 0.1.0
- Initial release with core Poisson disk sampling functionality
- Support for multi-dimensional spaces (2D, 3D, 4D, etc.)
- Symmetry operations (rotational, translational, custom)
- Hierarchical tiling with periodic boundary conditions
- Point selection based on interpolated data and thresholds
- Parallel processing support for large-scale tiling
- Comprehensive configuration system
- Utility functions for common operations

### Recent Changes
- Added comprehensive configuration system in `config.py`
- Enhanced symmetry operation support with invariant point detection
- Improved parallel processing for large-scale tiling
- Added utility functions for domain validation and optimal spacing estimation
- Enhanced error handling and validation throughout the codebase

### Known Limitations
- PointSelector currently optimized for 2D domains
- Symmetry operations work best with simple geometric transformations
- Large-scale tiling may require significant memory for very large domains
- Interpolation grid resolution affects PointSelector performance

### Future Enhancements
- Support for more complex symmetry groups
- Enhanced 3D+ visualization capabilities
- GPU acceleration for large-scale operations
- Additional point selection algorithms
- Integration with more scientific computing libraries 