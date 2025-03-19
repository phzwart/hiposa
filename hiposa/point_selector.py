import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay
import scipy.ndimage as ndi
from sklearn.model_selection import KFold

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
                 xy, 
                 levels, 
                 scales,
                 f_function, 
                 grid_x, 
                 grid_y, 
                 f_gt=None, 
                 tau=75.0, 
                 sign=1, 
                 eps=0.0, 
                 start_level=0, 
                 set_aside=0.5, 
                 lower=None, 
                 upper=None,
                 lower_tau_quantile=1.0,
                 n_splits_quantile=10,
                 border=4,
                 border_bias=0.5,
                 radius_scale=4.0,
                ):
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
        self.xy = xy
        self.levels = levels
        self.scales = scales
        self.f = f_function
        self.grid_x = grid_x
        self.grid_y = grid_y
        # Store grid shape for later use
        self.grid_shape = self.grid_x.shape
        self.f_gt = f_gt
        self.tau = tau
        self.sign = sign
        self.eps = eps
        self.start_level = start_level
        self.set_aside = set_aside
        self.lower = lower
        self.upper = upper
        self.lower_tau_quantile = lower_tau_quantile
        self.n_splits_quantile = n_splits_quantile
        self.border = border
        self.border_bias = max(0.0, min(1.0, border_bias))  # Clamp between 0 and 1
        self.radius_scale = max(1.0, radius_scale)  # Must be at least 1
        
        # Determine the threshold factor based on sign
        self.factor = 1 - eps if sign > 0 else 1 + eps
        
        # Initialize selection array
        self.sel = levels <= start_level
        self.index_array = np.arange(len(levels))

    def interpolate_sparse_data(self, these_xy, these_values, fill_value=np.nan):        
        # Create the interpolator
        interp = CloughTocher2DInterpolator(
            these_xy, 
            these_values, 
            fill_value=fill_value
        )
        surface = interp(self.grid_x, self.grid_y)

        hull = Delaunay(these_xy)
        grid_points = np.column_stack((self.grid_x.ravel(), self.grid_y.ravel()))
        mask = hull.find_simplex(grid_points) < 0
        surface.ravel()[mask] = np.nan 
        return surface



    @staticmethod
    def get_distance_transform(heatmap, threshold):
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
    def get_quantiles(data, tau, n_splits=5):
        """
        Calculate quantiles of the data using cross-validation.
        
        Parameters:
        -----------
        data : array-like
            Input data.
        tau : float
            Percentile value.
        n_splits : int, optional
            Number of folds for cross-validation, default is 5.
            
        Returns:
        --------
        list
            List of quantile values.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = []
        for _, idx_set in kf.split(data):
            tmp = data[idx_set]
            results.append(np.percentile(tmp, tau))
        return results
    
    def evaluate_points(self, points):
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
    
    def compute_threshold(self, work_xy, work_f_values, cal_f_values):
        """
        Compute the threshold based on working and calibration function values.
        
        Parameters:
        -----------
        work_xy : array-like
            Working set of (x, y) coordinates.
        work_f_values : array-like
            Function values at working coordinates.
        cal_f_values : array-like
            Function values at calibration coordinates.
            
        Returns:
        --------
        tuple
            Threshold value and intermediate values used in calculation.
        """
        # Interpolate function values to grid
        grid_values = self.interpolate_sparse_data(work_xy, work_f_values) #
        masker = np.ones_like(grid_values,  dtype=bool)
        masker[self.border:-self.border, self.border:-self.border] = False
        grid_values[masker]=np.nan
        isnan = np.isnan(grid_values)
        plt.imshow(grid_values)
        plt.show()
        print(masker.shape)

        # Calculate percentiles
        percentile_work_obs = np.percentile(work_f_values, self.tau)
        percentile_work = np.percentile(grid_values[~isnan], self.tau)
        
        # Calculate calibration percentile with delta adjustment
        qs = self.get_quantiles(cal_f_values, self.tau, n_splits=self.n_splits_quantile)
        low_delta = np.percentile(qs, 50) - np.percentile(qs, self.lower_tau_quantile )
        percentile_cal = np.percentile(cal_f_values, self.tau) - low_delta
        
        # Calculate deltas and threshold
        delta_g_o = abs(percentile_work - percentile_work_obs)
        if self.sign < 0:
            threshold = max(percentile_work, percentile_cal - self.sign*delta_g_o) * self.factor
        if self.sign > 0:
            threshold = min(percentile_work, percentile_cal + self.sign*delta_g_o) * self.factor
          
        return threshold, grid_values, percentile_work_obs, percentile_work, percentile_cal + delta_g_o

    
    def select_points_at_level(self, level, threshold, grid_values):
        """
        Select points at a specific level based on the threshold.
        
        Parameters:
        -----------
        level : int
            Level to select points from.
        threshold : float
            Threshold value for selection.
        grid_values : array-like
            Interpolated function values on the grid.
            
        Returns:
        --------
        np.ndarray
            Array of newly selected points.
        """
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
            
            # Check if point meets threshold criteria
            meets_threshold = (value > threshold if self.sign > 0 else value <= threshold)
            
            if meets_threshold:
                # If border bias is active, apply probability filter for non-border points
                if self.border_bias > 0:
                    is_border = distance_transform[grid_idx]
                    if not is_border:
                        # Point is not in border region, apply probability filter
                        if np.random.random() > (1 - self.border_bias):
                            continue  # Skip this point with probability border_bias
                
                if not self.sel[s]:
                    new_ones.append((tx, ty))
                self.sel[s] = True
            
        return np.array(new_ones)
    
    def plot_results(self, these_xy, new_ones, threshold, surface, mask, title=None, level=None):
        """
        Plot the results of the selection process.
        
        Parameters:
        -----------
        these_xy : array-like
            Currently selected (x, y) coordinates.
        new_ones : array-like
            Newly selected (x, y) coordinates.
        threshold : float
            Threshold value used for selection.
        surface : array-like
            Interpolated surface values.
        mask : array-like
            Boolean mask indicating areas above the threshold.
        title : str, optional
            Custom title for the plot. If None, uses default threshold title.
        level : int, optional
            Current level in the selection process. Used to display scale value.
        """
        plt.figure(figsize=(10, 8))
        
        # Set title
        if title is not None:
            plt.title(title)
        else:
            plt.title(f"Estimated threshold: {threshold:.3f}")
        
        # Plot surface
        im = plt.imshow(surface, origin="lower", extent=(self.lower, self.upper, self.lower, self.upper))
        plt.colorbar()
        
        # Add info box with threshold, points count, and scale if available
        text_lines = [f"Threshold: {threshold:.3f}"]
        text_lines.append(f"Total points: {len(these_xy)}")
        if new_ones is not None:
            text_lines.append(f"New points: {len(new_ones)}")
        if level is not None and level < len(self.scales):
            text_lines.append(f"Scale: {self.scales[level]:.3f}")
        
        text_str = "\n".join(text_lines)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # Place text box in upper left corner
        plt.text(0.02, 0.98, text_str, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
        
        # Plot existing points
        plt.scatter(these_xy[:, 0], these_xy[:, 1], marker="+", c="black", s=3)
        
        # Plot new points if any
        if new_ones is not None and new_ones.shape[0] > 0:
            plt.scatter(new_ones[:, 0], new_ones[:, 1], marker="+", c="red", s=3)
        
        # Plot threshold mask
        plt.imshow(mask, alpha=0.250, origin="lower", 
                  extent=(self.lower, self.upper, self.lower, self.upper), cmap="Reds")
        plt.show()

        
    
    def run(self, max_level=19):
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
            print(len(new_ones))
            
            # Print statistics
            print(f"Level {level}:")
            print(f"  Percentiles: work_obs={p_work_obs:.3f}, work={p_work:.3f}, cal+delta={p_cal_delta:.3f}")
            print(f"  Threshold: {threshold:.3f}")
            print(f"  New points: {new_ones.shape[0]}")
            
            # Plot results
            mask = surface > threshold
            self.plot_results(these_xy, new_ones, threshold, surface, mask, title=f"Level {level}", level=level)

        if self.f_gt is not None:
            new_threshold = np.percentile(self.f_gt.flatten(), self.tau)
            mask = self.f_gt > new_threshold
            surface = self.f_gt
            self.plot_results(these_xy, new_ones, new_threshold, surface, mask, title=f"Ground truth", level=None)

        return self.sel
