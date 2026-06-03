import logging
import numpy as np
from scipy.spatial import KDTree
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Callable, Union, Any
import numpy.typing as npt
import numbers

logger = logging.getLogger(__name__)

# Optional config integration
try:
    from .config import SAMPLING_CONFIG
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

class PoissonDiskSamplerWithExisting:
    """
    A class to generate samples using Poisson Disk Sampling within a specified domain,
    constrained by an existing set of points.

    This class implements the Poisson disk sampling algorithm with support for:
    - Multi-dimensional spaces (2D, 3D, 4D, etc.)
    - Symmetry operations (rotational, translational, etc.)
    - Periodic boundary conditions for tiling
    - Hierarchical sampling with existing point constraints

    Attributes:
        domain (np.ndarray): Boundaries for each dimension in the domain.
        r (float): Minimum distance between samples.
        k (int): Maximum number of attempts to generate a new sample around each existing sample.
        existing_points (Optional[np.ndarray]): Array of points that already exist in the domain.
        existing_labels (Optional[np.ndarray]): Array of labels corresponding to existing points.
        wrap (bool): Whether to use wrap-around edges for tiling.
        symmetry_operators (List[Callable]): List of symmetry operations to apply.
        dimensions (int): Number of dimensions in the domain.
        cell_size (float): Cell size for grid-based optimization.
        samples (List[np.ndarray]): List of generated sample points.
        labels (np.ndarray): Array of labels for all points.
        kdtree (Optional[KDTree]): KDTree for efficient nearest neighbor searches.
    """

    def __init__(self, 
                 domain: List[Tuple[float, float]], 
                 r: float, 
                 existing_points: Optional[np.ndarray] = None, 
                 existing_labels: Optional[np.ndarray] = None,
                 k: Optional[int] = None, 
                 symmetry_operators: Optional[List[Callable]] = None, 
                 wrap: bool = False) -> None:
        """
        Initialize the PoissonDiskSamplerWithExisting.

        Args:
            domain: List of (min, max) tuples defining boundaries for each dimension.
            r: Minimum distance between samples.
            existing_points: Array of pre-existing points to respect.
            existing_labels: Array of labels for pre-existing points.
            k: Maximum number of attempts to generate a new sample.
            symmetry_operators: List of symmetry operations to apply to points.
            wrap: Whether to use wrap-around edges for tiling.

        Raises:
            ValueError: If domain is empty or r is non-positive.
        """
        if not domain:
            raise ValueError("Domain cannot be empty")
        if r <= 0:
            raise ValueError("Minimum distance r must be positive")
            
        self.domain = np.array(domain)
        self.r = r
        
        # Use config default for k if not provided
        if USE_CONFIG and k is None:
            self.k = SAMPLING_CONFIG.DEFAULT_K
        else:
            self.k = k if k is not None else 60
        
        self.dimensions = len(domain)
        self.cell_size = r / np.sqrt(self.dimensions)
        self.existing_points = existing_points
        self.symmetry_operators = symmetry_operators if symmetry_operators is not None else []
        self.wrap = wrap

        if existing_points is not None:
            self.samples = existing_points.tolist()
            self.kdtree = KDTree(existing_points)
            self._indexed_count = len(existing_points)
            self.labels = existing_labels if existing_labels is not None else np.array(
                ["existing"] * len(existing_points))
        else:
            self.samples = []
            self.kdtree = None
            self._indexed_count = 0
            self.labels = np.array([])

    def _rebuild_kdtree(self) -> None:
        """Rebuild the KDTree so it indexes all current samples.

        The KDTree is used to accelerate non-periodic validity checks. It is an
        immutable structure, so it is rebuilt in bulk rather than updated
        incrementally; any samples added after the last rebuild are tracked via
        ``self._indexed_count`` and checked separately in :meth:`is_valid_point`.
        """
        if len(self.samples) > 0:
            self.kdtree = KDTree(np.asarray(self.samples))
            self._indexed_count = len(self.samples)

    def generate_points_around(self, point: np.ndarray) -> np.ndarray:
        """
        Generate potential points around a given sample within the allowed radius.

        Args:
            point: The point around which to generate new points.

        Returns:
            Array of new points around the given point.

        Raises:
            ValueError: If point is not in the correct dimension.
        """
        if len(point) != self.dimensions:
            raise ValueError(f"Point dimension {len(point)} does not match domain dimension {self.dimensions}")
            
        radius = np.sqrt(
            np.random.uniform(self.r ** 2, (2 * self.r) ** 2, self.k))
        directions = np.random.normal(0, 1, (self.k, self.dimensions))
        unit_vectors = directions / np.linalg.norm(directions, axis=1)[:, None]
        new_points = point + radius[:, None] * unit_vectors

        if self.wrap:
            # Apply wrap-around for each dimension with arbitrary bounds
            for dim in range(self.dimensions):
                min_bound, max_bound = self.domain[dim]
                new_points[:, dim] = (new_points[:, dim] - min_bound) % (
                            max_bound - min_bound) + min_bound

        return new_points

    def is_valid_point(self, point: np.ndarray) -> bool:
        """
        Validate a candidate point against the domain bounds and existing samples.

        For non-periodic domains this queries the KDTree (O(log n)) plus a small
        set of samples added since the last rebuild. For periodic domains
        (``wrap=True``) it falls back to a vectorized wrap-around distance check,
        since KDTree's box-periodicity assumptions do not match arbitrary domain
        bounds.

        Args:
            point: The point to validate.

        Returns:
            True if the point is valid, False otherwise.
        """
        point = np.asarray(point)

        # Check domain bounds
        if np.any(point < self.domain[:, 0]) or np.any(point >= self.domain[:, 1]):
            return False

        # If no existing points, any point within bounds is valid
        n_samples = len(self.samples)
        if n_samples == 0:
            return True

        if self.wrap:
            # Vectorized wrap-around distance calculation (brute-force fallback).
            points = np.asarray(self.samples)
            diff = np.abs(point - points)
            domain_size = self.domain[:, 1] - self.domain[:, 0]
            wrapped_diff = np.minimum(diff, domain_size - diff)
            distances = np.sqrt(np.sum(wrapped_diff ** 2, axis=1))
            return not np.any(distances < self.r)

        # Non-periodic: use the KDTree for the indexed prefix of samples.
        if self.kdtree is not None:
            dist, _ = self.kdtree.query(point, k=1)
            if dist < self.r:
                return False

        # Check any samples added since the last KDTree rebuild.
        if self._indexed_count < n_samples:
            pending = np.asarray(self.samples[self._indexed_count:])
            distances = np.sqrt(np.sum((point - pending) ** 2, axis=1))
            if np.any(distances < self.r):
                return False

        return True

    def check_orbit_validity(self, orbit: List[np.ndarray], epsilon: float = 1e-10) -> bool:
        """
        Check if all points in an orbit maintain proper distance relationships.
        Points must either be very close (< epsilon) or far enough apart (>= r).
        
        Args:
            orbit: List of points in the orbit.
            epsilon: Threshold for considering points identical.
            
        Returns:
            True if the orbit is valid, False otherwise.
        """
        for i in range(len(orbit)):
            for j in range(i + 1, len(orbit)):
                if self.wrap:
                    diff = np.abs(orbit[i] - orbit[j])
                    domain_size = self.domain[:, 1] - self.domain[:, 0]
                    wrapped_diff = np.minimum(diff, domain_size - diff)
                    dist = np.sqrt(np.sum(wrapped_diff ** 2))
                else:
                    dist = np.sqrt(np.sum((orbit[i] - orbit[j]) ** 2))
                
                # Distance must be either very small (~ identical points)
                # or larger than minimum spacing
                if dist >= epsilon and dist < self.r:
                    return False
        return True

    def apply_symmetry(self, point: np.ndarray) -> Optional[List[np.ndarray]]:
        """
        Applies all symmetry operations to a point and returns the complete orbit.
        Only returns the orbit if all points in it maintain proper distance relationships.

        Args:
            point: The point to which symmetry operations are applied.

        Returns:
            A list of points in the complete orbit under the symmetry operators,
            or None if the orbit is invalid.
        """
        symmetric_points = [np.array(point)]
        epsilon = 1e-10  # Threshold for considering points identical
        
        # For each operator
        for op in self.symmetry_operators:
            current_orbit = symmetric_points.copy()
            # Apply operator to all points in current orbit
            for base_point in current_orbit:
                current_point = base_point
                while True:
                    transformed = op(current_point)
                    if transformed is None:
                        break
                        
                    # Check if transformed point is close to any existing point in orbit
                    is_new = True
                    for existing in symmetric_points:
                        if self.wrap:
                            diff = np.abs(transformed - existing)
                            domain_size = self.domain[:, 1] - self.domain[:, 0]
                            wrapped_diff = np.minimum(diff, domain_size - diff)
                            dist = np.sqrt(np.sum(wrapped_diff ** 2))
                        else:
                            dist = np.sqrt(np.sum((transformed - existing) ** 2))
                        if dist < epsilon:
                            is_new = False
                            break
                    
                    if not is_new:
                        break
                        
                    symmetric_points.append(transformed)
                    current_point = transformed
        
        # Check if the complete orbit is valid
        if self.check_orbit_validity(symmetric_points, epsilon):
            return symmetric_points
        return None

    def find_invariant_points(self, operator: Callable) -> List[np.ndarray]:
        """
        Find points that are invariant (fixed points) under a symmetry operator
        using Nelder-Mead optimization to minimize the distance between a point
        and its transform.
        
        Args:
            operator: The symmetry operator function.
            
        Returns:
            List of invariant points found within the domain.
        """
        epsilon = 1e-10
        
        def objective(point: np.ndarray) -> float:
            """Distance between point and its transform."""
            point = np.array(point)
            transformed = operator(point)
            if transformed is None:
                return float('inf')
            
            if self.wrap:
                diff = np.abs(point - transformed)
                domain_size = self.domain[:, 1] - self.domain[:, 0]
                wrapped_diff = np.minimum(diff, domain_size - diff)
                return np.sum(wrapped_diff ** 2)
            else:
                return np.sum((point - transformed) ** 2)
        
        # Try multiple starting points to find all possible invariant points
        invariant_points = []
        n_attempts = 10  # Number of random starting points
        
        for _ in range(n_attempts):
            # Random starting point within domain
            x0 = np.random.uniform(self.domain[:, 0], self.domain[:, 1])
            
            # Minimize distance between point and its transform
            result = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'xatol': epsilon, 'fatol': epsilon}
            )
            
            if result.fun < epsilon:  # Found an invariant point
                point = result.x
                
                # Ensure point is within domain and wrap if needed
                if self.wrap:
                    for dim in range(self.dimensions):
                        min_bound, max_bound = self.domain[dim]
                        point[dim] = (point[dim] - min_bound) % (
                            max_bound - min_bound) + min_bound
                else:
                    # Clip to domain bounds
                    point = np.clip(point, self.domain[:, 0], self.domain[:, 1])
                
                # Check if this is a new point
                is_new = True
                for existing in invariant_points:
                    if self.wrap:
                        diff = np.abs(point - existing)
                        domain_size = self.domain[:, 1] - self.domain[:, 0]
                        wrapped_diff = np.minimum(diff, domain_size - diff)
                        dist = np.sqrt(np.sum(wrapped_diff ** 2))
                    else:
                        dist = np.sqrt(np.sum((point - existing) ** 2))
                    if dist < epsilon:
                        is_new = False
                        break
                
                if is_new:
                    invariant_points.append(point)
        
        return invariant_points

    def sample(self, new_label: Optional[Union[int, str]] = None, return_new_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a sample of points using the Poisson Disk Sampling method.
        Now includes invariant points under symmetry operators.
        """
        if new_label is None:
            new_label = 0
            if len(self.labels) > 0:
                logger.debug("self.labels type: %s contents: %s", type(self.labels), self.labels)
                try:
                    numeric_labels = []
                    for label in self.labels:
                        if isinstance(label, numbers.Number):
                            numeric_labels.append(int(label))
                        elif isinstance(label, str) and label.isdigit():
                            numeric_labels.append(int(label))
                        else:
                            numeric_labels.append(0)
                    new_label = max(numeric_labels) + 1
                except (ValueError, TypeError):
                    new_label = len(self.labels)
            logger.debug("Computed new_label = %s", new_label)

        if not self.samples:
            # First try to add invariant points
            for op in self.symmetry_operators:
                invariant_points = self.find_invariant_points(op)
                for point in invariant_points:
                    if self.is_valid_point(point):
                        self.samples.append(point)
                        self.labels = np.append(self.labels, 
                            new_label if new_label is not None else "new")
            
            # If no points added yet, find a valid initial orbit
            if not self.samples:
                while True:
                    initial_point = np.random.uniform(self.domain[:, 0],
                                                   self.domain[:, 1],
                                                   self.dimensions)
                    symmetric_points = self.apply_symmetry(initial_point)
                    if symmetric_points is not None:
                        self.samples.extend(symmetric_points)
                        self.labels = np.concatenate((self.labels, np.array(
                            [new_label if new_label is not None else "new"] * len(
                                symmetric_points))))
                        break
            
            active_list = list(range(len(self.samples)))
        else:
            active_list = list(range(len(self.samples)))

        new_points = []
        new_labels = []

        # Update KDTree periodically
        update_frequency = 10
        points_since_update = 0
        
        # First pass with original k
        while active_list:
            i = np.random.choice(active_list)
            current_point = self.samples[i]
            generated_points = self.generate_points_around(current_point)

            valid_found = False
            for point in generated_points:
                # Apply symmetry and check if orbit is valid
                symmetric_points = self.apply_symmetry(point)
                if symmetric_points is not None:
                    # Check if all points in the orbit are valid with existing points
                    all_valid = all(self.is_valid_point(p) for p in symmetric_points)
                    
                    if all_valid:
                        # Add all points from the valid orbit
                        for sym_point in symmetric_points:
                            self.samples.append(sym_point)
                            new_index = len(self.samples) - 1
                            label = new_label if new_label is not None else "new"
                            self.labels = np.append(self.labels, label)
                            active_list.append(new_index)
                            new_points.append(sym_point)
                            new_labels.append(label)
                        valid_found = True
                        break

            if not valid_found:
                active_list.remove(i)

            points_since_update += 1
            if points_since_update >= update_frequency:
                self._rebuild_kdtree()
                points_since_update = 0

        # Second pass with increased k
        original_k = self.k
        self.k = 80  # Increase k for second pass
        
        # Reset active list for second pass
        active_list = list(range(len(self.samples)))
        
        while active_list:
            i = np.random.choice(active_list)
            current_point = self.samples[i]
            generated_points = self.generate_points_around(current_point)

            valid_found = False
            for point in generated_points:
                # Apply symmetry and check if orbit is valid
                symmetric_points = self.apply_symmetry(point)
                if symmetric_points is not None:
                    # Check if all points in the orbit are valid with existing points
                    all_valid = all(self.is_valid_point(p) for p in symmetric_points)
                    
                    if all_valid:
                        # Add all points from the valid orbit
                        for sym_point in symmetric_points:
                            self.samples.append(sym_point)
                            new_index = len(self.samples) - 1
                            label = new_label if new_label is not None else "new"
                            self.labels = np.append(self.labels, label)
                            active_list.append(new_index)
                            new_points.append(sym_point)
                            new_labels.append(label)
                        valid_found = True
                        break

            if not valid_found:
                active_list.remove(i)

            points_since_update += 1
            if points_since_update >= update_frequency:
                self._rebuild_kdtree()
                points_since_update = 0

        # Restore original k
        self.k = original_k

        logger.debug("Final labels array = %s", self.labels)
        if return_new_only:
            return np.array(new_points), np.array(new_labels)
        else:
            return np.array(self.samples), self.labels
