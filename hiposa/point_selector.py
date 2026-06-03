import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import scipy.ndimage as ndi
from sklearn.model_selection import KFold
from typing import List, Tuple, Optional, Union, Any, Callable
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Optional config integration
try:
    from .config import POINT_SELECTOR_CONFIG
    USE_CONFIG = True
except ImportError:
    USE_CONFIG = False

class PointSelector:
    """
    A class for selecting points based on thresholds calculated from interpolated data.
    
    This class implements a hierarchical point selection algorithm that:
    1. Interpolates sparse data points to a regular grid
    2. Computes thresholds based on working and calibration sets
    3. Selects points at different levels based on these thresholds
    4. Visualizes the selection process
    
    Parameters
    ----------
    xy : array-like
        Array of (x, y) coordinates.
    levels : array-like
        Array of level values corresponding to each coordinate in xy.
    scales : array-like
        Array of scale values for each level.
    f_function : callable
        Function to evaluate at each coordinate.
    grid_x : array-like
        Grid of x coordinates for interpolation.
    grid_y : array-like
        Grid of y coordinates for interpolation.
    f_gt : array-like
        Ground truth function values on the grid.
    tau : float, optional
        Percentile value to use for thresholding, default is 75.0.
    sign : int, optional
        Direction of thresholding (1 for greater than, -1 for less than), default is 1.
    eps : float, optional
        Small adjustment to the threshold, default is 0.0.
    start_level : int, optional
        Level to start the selection process, default is 0.
    set_aside : float, optional
        Fraction of points to set aside for calibration, default is 0.5.
    lower : float, optional
        Lower bound for plotting, default is None.
    upper : float, optional
        Upper bound for plotting, default is None.
    lower_tau_quantile : float, optional
        Lower quantile for threshold adjustment, default is 1.0.
    n_splits_quantile : int, optional
        Number of splits for cross-validation in quantile calculation, default is 10.
    border : int, optional
        Border size for masking, default is 4.
    border_bias : float, optional
        Value between 0 and 1 controlling bias towards sampling border regions.
        0 means no bias, 1 means maximum bias towards borders. Default is 0.5.
    radius_scale : float, optional
        Multiplier for the effective radius when considering border regions.
        Higher values mean larger border regions are considered. Default is 4.0.
    """
    
    def __init__(self, 
                 xy: npt.NDArray, 
                 levels: npt.NDArray, 
                 scales: npt.NDArray,
                 f_function: Callable, 
                 grid_x: npt.NDArray, 
                 grid_y: npt.NDArray, 
                 f_gt: Optional[npt.NDArray] = None, 
                 tau: Optional[float] = None, 
                 sign: Optional[int] = None, 
                 eps: Optional[float] = None, 
                 start_level: Optional[int] = None, 
                 set_aside: Optional[float] = None, 
                 lower: Optional[float] = None, 
                 upper: Optional[float] = None,
                 lower_tau_quantile: Optional[float] = None,
                 n_splits_quantile: Optional[int] = None,
                 border: Optional[int] = None,
                 border_bias: Optional[float] = None,
                 radius_scale: Optional[float] = None,
                ) -> None:
        """
        Initialize the PointSelector with the necessary data and parameters.
        
        Parameters:
        -----------
        xy : array-like
            Array of (x, y) coordinates.
        levels : array-like
            Array of level values corresponding to each coordinate in xy.
        scales : array-like
            Array of scale values for each level.
        f_function : callable
            Function to evaluate at each coordinate.
        grid_x : array-like
            Grid of x coordinates for interpolation.
        grid_y : array-like
            Grid of y coordinates for interpolation.
        f_gt : array-like, optional
            Ground truth function values on the grid.
        tau : float, optional
            Percentile value to use for thresholding, default is 75.0.
        sign : int, optional
            Direction of thresholding (1 for greater than, -1 for less than), default is 1.
        eps : float, optional
            Small adjustment to the threshold, default is 0.0.
        start_level : int, optional
            Level to start the selection process, default is 0.
        set_aside : float, optional
            Fraction of points to set aside for calibration, default is 0.5.
        lower : float, optional
            Lower bound for plotting, default is None.
        upper : float, optional
            Upper bound for plotting, default is None.
        lower_tau_quantile : float, optional
            Lower quantile for threshold adjustment, default is 1.0.
        n_splits_quantile : int, optional
            Number of splits for cross-validation in quantile calculation, default is 10.
        border : int, optional
            Border size for masking, default is 4.
        border_bias : float, optional
            Value between 0 and 1 controlling bias towards sampling border regions.
            0 means no bias, 1 means maximum bias towards borders. Default is 0.5.
        radius_scale : float, optional
            Multiplier for the effective radius when considering border regions.
            Higher values mean larger border regions are considered. Default is 4.0.
        """
        # Raise ValueError if any required array is empty
        if (xy is None or xy.size == 0 or
            levels is None or levels.size == 0 or
            scales is None or scales.size == 0 or
            grid_x is None or grid_x.size == 0 or
            grid_y is None or grid_y.size == 0 or
            (f_gt is not None and f_gt.size == 0)):
            raise ValueError("All input arrays (xy, levels, scales, grid_x, grid_y) must be non-empty.")
        # Check that xy, levels, and scales all have the same length
        if not (len(xy) == len(levels) == len(scales)):
            raise ValueError("xy, levels, and scales must all have the same length.")
        self.xy = xy
        self.levels = levels
        self.scales = scales
        self.f = f_function
        self.grid_x = grid_x
        self.grid_y = grid_y
        # Store grid shape for later use
        self.grid_shape = self.grid_x.shape
        self.f_gt = f_gt
        
        # Use config defaults if available and parameter not provided
        if USE_CONFIG:
            self.tau = tau if tau is not None else POINT_SELECTOR_CONFIG.DEFAULT_TAU
            self.sign = sign if sign is not None else POINT_SELECTOR_CONFIG.DEFAULT_SIGN
            self.eps = eps if eps is not None else POINT_SELECTOR_CONFIG.DEFAULT_EPS
            self.start_level = start_level if start_level is not None else POINT_SELECTOR_CONFIG.DEFAULT_START_LEVEL
            self.set_aside = set_aside if set_aside is not None else POINT_SELECTOR_CONFIG.DEFAULT_SET_ASIDE
            self.lower_tau_quantile = lower_tau_quantile if lower_tau_quantile is not None else POINT_SELECTOR_CONFIG.DEFAULT_LOWER_TAU_QUANTILE
            self.n_splits_quantile = n_splits_quantile if n_splits_quantile is not None else POINT_SELECTOR_CONFIG.DEFAULT_N_SPLITS_QUANTILE
            self.border = border if border is not None else POINT_SELECTOR_CONFIG.DEFAULT_BORDER
            self.border_bias = border_bias if border_bias is not None else POINT_SELECTOR_CONFIG.DEFAULT_BORDER_BIAS
            self.radius_scale = radius_scale if radius_scale is not None else POINT_SELECTOR_CONFIG.DEFAULT_RADIUS_SCALE
        else:
            # Fallback to hardcoded defaults
            self.tau = tau if tau is not None else 75.0
            self.sign = sign if sign is not None else 1
            self.eps = eps if eps is not None else 0.0
            self.start_level = start_level if start_level is not None else 0
            self.set_aside = set_aside if set_aside is not None else 0.5
            self.lower_tau_quantile = lower_tau_quantile if lower_tau_quantile is not None else 1.0
            self.n_splits_quantile = n_splits_quantile if n_splits_quantile is not None else 10
            self.border = border if border is not None else 4
            self.border_bias = border_bias if border_bias is not None else 0.5
            self.radius_scale = radius_scale if radius_scale is not None else 4.0
        
        self.lower = lower
        self.upper = upper
        
        # Clamp values to valid ranges
        self.border_bias = max(0.0, min(1.0, self.border_bias))  # Clamp between 0 and 1
        self.radius_scale = max(1.0, self.radius_scale)  # Must be at least 1
        
        # Determine the threshold factor based on sign
        if sign is None:
            sign = 1  # Default to positive
        if eps is None:
            eps = 0.0  # Default eps value
        self.factor = 1 - eps if sign > 0 else 1 + eps
        
        # Initialize selection array
        if start_level is None:
            start_level = 0  # Default start_level value
        self.sel = levels <= start_level
        self.index_array = np.arange(len(levels))

    def interpolate_sparse_data(self, these_xy: npt.NDArray, these_values: npt.NDArray, fill_value: float = np.nan) -> npt.NDArray:
        """
        Interpolate sparse data points to a regular grid.
        """
        # If not enough points, return fill_value grid
        if these_xy.shape[0] < 3:
            return np.full((len(self.grid_x), len(self.grid_y)), fill_value)
        try:
            interpolator = CloughTocher2DInterpolator(these_xy, these_values, fill_value=fill_value)
            grid_xx, grid_yy = np.meshgrid(self.grid_x, self.grid_y, indexing='ij')
            result = interpolator(grid_xx, grid_yy)
            # Ensure result is 2D
            if result.ndim == 1:
                result = result.reshape(grid_xx.shape)
            return result
        except Exception:
            # Fallback: return fill_value grid
            return np.full((len(self.grid_x), len(self.grid_y)), fill_value)

    @staticmethod
    def get_distance_transform(heatmap: npt.NDArray, threshold: float) -> npt.NDArray:
        """
        Compute distance transform from binary mask created by thresholding.
        
        Args:
            heatmap: Input heatmap array.
            threshold: Threshold value for creating binary mask.
            
        Returns:
            Distance transform array.
        """
        binary_mask = heatmap > threshold
        dilated = ndi.binary_dilation(binary_mask)
        eroded = ndi.binary_erosion(binary_mask)
        border_pixels = dilated ^ eroded  # XOR to get only the border
        D_chessboard = ndi.distance_transform_cdt(~border_pixels, metric='chessboard')
        D_manhattan = ndi.distance_transform_cdt(~border_pixels, metric='taxicab')
        # Weighted combination
        approx_euclidean = 0.707 * D_chessboard + 0.293 * D_manhattan
        return approx_euclidean
        
    @staticmethod
    def get_quantiles(data: npt.NDArray, tau: float, n_splits: int = 5) -> List[float]:
        """
        Calculate quantiles of the data using cross-validation.
        """
        # Handle case where data is empty
        if len(data) == 0:
            return [0.0]
        # Handle case where we have fewer samples than splits
        if len(data) < n_splits:
            return [np.percentile(data, tau)]
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        for _, idx_set in kf.split(data):
            tmp = data[idx_set]
            results.append(np.percentile(tmp, tau))
        return results

    def evaluate_points(self, points: npt.NDArray) -> npt.NDArray:
        """
        Evaluate the function at the given points.
        
        Parameters:
        -----------
        points : array-like
            Array of (x, y) coordinates.
            
        Returns:
        --------
        np.ndarray
            Array of function values.
        """
        return np.array([self.f(point) for point in points])
    
    def compute_threshold(self, work_xy: npt.NDArray, work_f_values: npt.NDArray, cal_f_values: npt.NDArray) -> Tuple[float, npt.NDArray, float, float, float]:
        """
        Compute the threshold based on working and calibration function values.
        """
        isnan = np.isnan(work_f_values) if hasattr(work_f_values, 'dtype') else []
        # Handle empty arrays
        if len(work_f_values) == 0:
            return 0.0, np.array([]), 0.0, 0.0, 0.0
        percentile_work_obs = np.percentile(work_f_values, self.tau)
        # grid_values is computed from interpolate_sparse_data
        grid_values = self.interpolate_sparse_data(work_xy, work_f_values)
        isnan_grid = np.isnan(grid_values)
        valid_grid_values = grid_values[~isnan_grid]
        if len(valid_grid_values) == 0:
            percentile_work = np.percentile(work_f_values, self.tau)
        else:
            percentile_work = np.percentile(valid_grid_values, self.tau)
        # Handle empty cal_f_values
        if len(cal_f_values) == 0:
            qs = [0.0]
            low_delta = 0.0
            percentile_cal = 0.0
        else:
            qs = self.get_quantiles(cal_f_values, self.tau, n_splits=self.n_splits_quantile)
            low_delta = np.percentile(qs, 50) - np.percentile(qs, self.lower_tau_quantile)
            percentile_cal = np.percentile(cal_f_values, self.tau) - low_delta
        delta_g_o = abs(percentile_work - percentile_work_obs)
        if self.sign < 0:
            threshold = max(percentile_work, percentile_cal - self.sign*delta_g_o) * self.factor
        if self.sign > 0:
            threshold = min(percentile_work, percentile_cal + self.sign*delta_g_o) * self.factor
        return threshold, grid_values, percentile_work_obs, percentile_work, percentile_cal + delta_g_o

    def select_points_at_level(self, level: int, threshold: float, grid_values: npt.NDArray) -> npt.NDArray:
        """
        Select points at a specific level based on the threshold.
        """
        # If grid_values is empty, return empty array
        if grid_values.size == 0:
            return np.empty((0, 2))
        sel_2 = self.levels == level
        next_xy = self.xy[sel_2]
        sel_2 = self.index_array[sel_2]
        new_ones = []
        # Compute distance transform if border bias is active
        if self.border_bias > 0:
            distance_transform = self.get_distance_transform(grid_values, threshold)
            # Scale distances by radius_scale
            distance_transform = distance_transform <= self.scales[level-1] * self.radius_scale
        for s, this_next_one in zip(sel_2, next_xy):
            tx, ty = this_next_one
            d = np.sqrt((self.grid_x - tx)**2 + (self.grid_y - ty)**2)
            indx = np.argmin(d)
            grid_idx = np.unravel_index(indx, self.grid_shape)
            value = grid_values[grid_idx]
            # Ensure value is a scalar
            if isinstance(value, np.ndarray):
                value = value.item() if value.size == 1 else value.ravel()[0]
            else:
                value = float(value)
            # Skip if value is nan
            if np.isnan(value):
                continue
            # Check if point meets threshold criteria
            meets_threshold = (value > threshold if self.sign > 0 else value <= threshold)
            if meets_threshold:
                # If border bias is active, apply probability filter for non-border points
                if self.border_bias > 0:
                    is_border = distance_transform[grid_idx]
                    # Ensure is_border is a scalar
                    if isinstance(is_border, np.ndarray):
                        is_border = is_border.item() if is_border.size == 1 else bool(is_border.ravel()[0])
                    else:
                        is_border = bool(is_border)
                    if not is_border:
                        # Point is not in border region, apply probability filter
                        if np.random.random() > (1 - self.border_bias):
                            continue  # Skip this point with probability border_bias
                if not self.sel[s]:
                    new_ones.append((tx, ty))
                self.sel[s] = True
        # Always return a consistently-shaped (N, 2) array, even when empty.
        if not new_ones:
            return np.empty((0, 2))
        return np.array(new_ones)
    
    def plot_results(self, these_xy: npt.NDArray, new_ones: npt.NDArray, threshold: float, surface: npt.NDArray, mask: npt.NDArray, title: Optional[str] = None, level: Optional[int] = None) -> None:
        """
        Plot the results of the selection process.
        """
        plt.figure(figsize=(10, 8))
        # Set title
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Level {level} (Threshold: {threshold:.3f})")
        # Only plot surface if it is valid
        if surface is not None and surface.size > 0 and np.any(np.isfinite(surface)):
            try:
                im = plt.imshow(surface, origin="lower", extent=(self.lower, self.upper, self.lower, self.upper))
                plt.colorbar(im, label="Interpolated Value")
            except Exception:
                pass  # Skip plotting if surface is invalid
        # Plot selected points
        if these_xy is not None and len(these_xy) > 0:
            plt.scatter(these_xy[:, 0], these_xy[:, 1], c="blue", label="Selected", s=40)
        if new_ones is not None and len(new_ones) > 0:
            plt.scatter(new_ones[:, 0], new_ones[:, 1], c="red", label="New", s=60, marker="*")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.close()  # Close to avoid GUI issues in tests

    def run(self, max_level: int = 19) -> npt.NDArray:
        """
        Run the point selection algorithm.
        
        Parameters:
        -----------
        max_level : int, optional
            Maximum level to consider, default is 19.
            
        Returns:
        --------
        np.ndarray
            Boolean array indicating selected points.
        """
        these_xy = np.empty((0, 2))
        new_ones = np.empty((0, 2))
        for level in range(self.start_level + 1, max_level):
            # Get currently selected points
            these_xy = self.xy[self.sel]
            
            # Evaluate function at selected points
            f_values = self.evaluate_points(these_xy)
            
            # Split into working and calibration sets
            indices = np.arange(len(f_values))
            rnd_numb = np.random.uniform(0, 1, indices.shape)
            work_sel = rnd_numb > self.set_aside
            
            work_xy = these_xy[work_sel]
            work_f_values = f_values[work_sel]
            
            cal_xy = these_xy[~work_sel]
            cal_f_values = f_values[~work_sel]
            
            # Compute threshold and get interpolated grid values
            threshold, grid_values, p_work_obs, p_work, p_cal_delta = self.compute_threshold(
                work_xy, work_f_values, cal_f_values)
            
            # Create mask based on threshold
            surface = self.interpolate_sparse_data(these_xy, f_values)
            #surface = griddata(these_xy, f_values, (self.grid_x, self.grid_y), method=self.method)

            distance_map = self.get_distance_transform(surface, threshold)
            # Select points at current level
            tl = min(level, len(self.scales)-1)
            new_ones = self.select_points_at_level(level, threshold, grid_values) #, distance_map, self.scales[tl-1]*4 )

            logger.info("Level %s:", level)
            logger.info(
                "  Percentiles: work_obs=%.3f, work=%.3f, cal+delta=%.3f",
                p_work_obs, p_work, p_cal_delta,
            )
            logger.info("  Threshold: %.3f", threshold)
            logger.info("  New points: %s", new_ones.shape[0])
            
            # Plot results
            mask = surface > threshold
            self.plot_results(these_xy, new_ones, threshold, surface, mask, title=f"Level {level}", level=level)

        if self.f_gt is not None:
            new_threshold = np.percentile(self.f_gt.flatten(), self.tau)
            mask = self.f_gt > new_threshold
            surface = self.f_gt
            self.plot_results(these_xy, new_ones, new_threshold, surface, mask, title=f"Ground truth", level=None)

        return self.sel
