"""
Hyperplane visualization and analysis module for dskit.
Provides comprehensive hyperplane support for 2D and 3D visualizations with ML model integration.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Union, List, Tuple, Optional
import warnings

class Hyperplane:
    """
    A general hyperplane representation for 2D and 3D:
    2D: w1*x + w2*y + b = 0
    3D: w1*x + w2*y + w3*z + b = 0
    
    Supports visualization, distance calculation, and classification.
    """

    def __init__(self, weights: Union[List, np.ndarray], bias: float):
        """
        Initialize hyperplane with weights and bias.
        
        Parameters:
        -----------
        weights : array-like
            Weight vector for the hyperplane
        bias : float
            Bias term for the hyperplane
        """
        self.w = np.array(weights, dtype=float)
        self.b = float(bias)

        if len(self.w) not in [2, 3]:
            raise ValueError("Hyperplane supports only 2D (2 weights) or 3D (3 weights).")

    def equation(self) -> str:
        """
        Returns the hyperplane equation as a string.
        
        Returns:
        --------
        str
            Mathematical equation representation
        """
        if len(self.w) == 2:
            return f"{self.w[0]:.3f}*x + {self.w[1]:.3f}*y + {self.b:.3f} = 0"
        else:
            return f"{self.w[0]:.3f}*x + {self.w[1]:.3f}*y + {self.w[2]:.3f}*z + {self.b:.3f} = 0"

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns sign(w·x + b) for classification.
        
        Parameters:
        -----------
        X : array-like
            Input data points
            
        Returns:
        --------
        np.ndarray
            Predicted classes (+1 or -1)
        """
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return np.sign(X @ self.w + self.b)

    def distance(self, point: Union[List, np.ndarray]) -> float:
        """
        Perpendicular distance of a point from the hyperplane.
        
        Parameters:
        -----------
        point : array-like
            Point coordinates
            
        Returns:
        --------
        float
            Distance from hyperplane
        """
        point = np.array(point)
        numerator = abs(self.w @ point + self.b)
        denominator = np.linalg.norm(self.w)
        return numerator / denominator

    def plot_2d(self, x_range: Tuple[float, float] = (-5, 5), 
                X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                figsize: Tuple[int, int] = (10, 6), show_margin: bool = False,
                margin_distance: float = 1.0) -> None:
        """
        Plot 2D hyperplane with optional data points and margin visualization.
        
        Parameters:
        -----------
        x_range : tuple, default (-5, 5)
            Range for x-axis
        X : array-like, optional
            Data points to plot
        y : array-like, optional
            Labels for data points
        figsize : tuple, default (10, 6)
            Figure size
        show_margin : bool, default False
            Whether to show margin lines
        margin_distance : float, default 1.0
            Distance for margin lines
        """
        if len(self.w) != 2:
            raise ValueError("plot_2d() is only for 2D hyperplanes.")

        plt.figure(figsize=figsize)
        
        w1, w2 = self.w
        b = self.b

        # Plot the hyperplane
        if abs(w2) < 1e-10:  # Vertical line case
            x_line = -b / w1
            plt.axvline(x_line, color='red', linewidth=2, label="Decision Boundary")
        else:
            xs = np.linspace(*x_range, 100)
            ys = -(w1 * xs + b) / w2
            plt.plot(xs, ys, 'r-', linewidth=2, label="Decision Boundary")
            
            # Show margin lines if requested
            if show_margin:
                margin_b_pos = b - margin_distance * np.linalg.norm(self.w)
                margin_b_neg = b + margin_distance * np.linalg.norm(self.w)
                ys_margin_pos = -(w1 * xs + margin_b_pos) / w2
                ys_margin_neg = -(w1 * xs + margin_b_neg) / w2
                plt.plot(xs, ys_margin_pos, 'r--', alpha=0.5, label="Margin +")
                plt.plot(xs, ys_margin_neg, 'r--', alpha=0.5, label="Margin -")

        # Plot data points if provided
        if X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
            unique_classes = np.unique(y)
            colors = ['blue', 'orange', 'green', 'purple']
            
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                plt.scatter(X[mask, 0], X[mask, 1], 
                           c=colors[i % len(colors)], 
                           alpha=0.7, s=50, 
                           label=f'Class {cls}')

        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title(f"2D Hyperplane: {self.equation()}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def plot_3d(self, range_val: Tuple[float, float] = (-5, 5), 
                X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot 3D hyperplane with optional data points.
        
        Parameters:
        -----------
        range_val : tuple, default (-5, 5)
            Range for all axes
        X : array-like, optional
            Data points to plot
        y : array-like, optional
            Labels for data points
        figsize : tuple, default (10, 8)
            Figure size
        """
        if len(self.w) != 3:
            raise ValueError("plot_3d() is only for 3D hyperplanes.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        w1, w2, w3 = self.w
        b = self.b

        if abs(w3) < 1e-10:
            raise ValueError("Cannot solve for z when w3 ≈ 0")

        # Create mesh for plane
        xx = np.linspace(*range_val, 20)
        yy = np.linspace(*range_val, 20)
        xx, yy = np.meshgrid(xx, yy)
        zz = -(w1 * xx + w2 * yy + b) / w3

        # Plot the plane
        ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')

        # Plot data points if provided
        if X is not None and y is not None:
            X = np.array(X)
            y = np.array(y)
            unique_classes = np.unique(y)
            colors = ['blue', 'orange', 'green', 'purple']
            
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                          c=colors[i % len(colors)], 
                          alpha=0.7, s=50, 
                          label=f'Class {cls}')

        ax.set_xlabel("Feature 1", fontsize=12)
        ax.set_ylabel("Feature 2", fontsize=12)
        ax.set_zlabel("Feature 3", fontsize=12)
        ax.set_title(f"3D Hyperplane: {self.equation()}", fontsize=14)
        if X is not None and y is not None:
            ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_decision_regions(self, X: np.ndarray, y: np.ndarray, 
                             resolution: int = 100, 
                             figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Plot 2D decision regions with color-coded areas.
        
        Parameters:
        -----------
        X : array-like
            Data points (2D only)
        y : array-like
            Labels
        resolution : int, default 100
            Resolution for decision region mesh
        figsize : tuple, default (10, 6)
            Figure size
        """
        if len(self.w) != 2:
            raise ValueError("Decision regions only supported for 2D hyperplanes.")
            
        X = np.array(X)
        y = np.array(y)
        
        plt.figure(figsize=figsize)
        
        # Create mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                            np.linspace(y_min, y_max, resolution))
        
        # Get predictions for mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot decision regions
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        plt.contour(xx, yy, Z, colors='black', linewidths=0.5)
        
        # Plot data points
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       alpha=0.8, s=60, 
                       label=f'Class {cls}',
                       edgecolors='black')
        
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title("Decision Regions", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


class HyperplaneExtractor:
    """
    Extract and visualize hyperplanes from ML algorithms that produce linear decision boundaries.
    Supports: SVM, Logistic Regression, Perceptron, Linear Regression, LDA, and more.
    """

    def __init__(self, model):
        """
        Initialize with a trained ML model.
        
        Parameters:
        -----------
        model : sklearn model
            Trained model with linear decision boundary
        """
        self.model = model
        self.model_type = model.__class__.__name__
        self.w, self.b = self._extract_params(model)
        
        # Handle 1D case by creating a dummy hyperplane for plotting purposes
        if len(self.w) == 1:
            # For 1D regression, we don't create a Hyperplane object
            self.hyperplane = None
            self.is_1d = True
        else:
            self.hyperplane = Hyperplane(self.w, self.b)
            self.is_1d = False

    def _extract_params(self, model) -> Tuple[np.ndarray, float]:
        """
        Extract weights and bias from various ML model types.
        
        Parameters:
        -----------
        model : sklearn model
            Trained ML model
            
        Returns:
        --------
        tuple
            (weights, bias)
        """
        model_name = model.__class__.__name__

        # Standard linear models with coef_ and intercept_
        if hasattr(model, "coef_") and hasattr(model, "intercept_"):
            coef = model.coef_
            intercept = model.intercept_
            
            # Handle different coefficient shapes
            if coef.ndim > 1:
                if coef.shape[0] == 1:  # Single class case
                    w = coef[0]
                else:  # Multi-class case - use first class
                    w = coef[0]
                    warnings.warn(f"Multi-class model detected. Using first class coefficients.")
            else:
                w = coef
            
            # Handle intercept
            if np.isscalar(intercept):
                b = float(intercept)
            else:
                b = float(intercept[0])
                
            return np.array(w), b

        # GaussianNB - approximate linear boundary
        elif model_name == "GaussianNB":
            if len(model.classes_) != 2:
                raise ValueError("GaussianNB hyperplane extraction only supports binary classification")
            
            mean_diff = model.theta_[0] - model.theta_[1]
            # Handle potential division by zero
            var = model.var_[0] + 1e-10  # Add small epsilon
            w = mean_diff / var
            b = -0.5 * (np.sum(model.theta_[0] ** 2 / var) - np.sum(model.theta_[1] ** 2 / var))
            return w, b

        # Decision Tree - approximate with feature importance
        elif model_name in ["DecisionTreeClassifier", "DecisionTreeRegressor"]:
            warnings.warn("Decision Tree hyperplane is an approximation based on feature importance.")
            importances = model.feature_importances_
            # Create pseudo-linear boundary
            w = importances / np.sum(importances)
            b = 0.0
            return w, b

        else:
            raise ValueError(f"Model type '{model_name}' not supported for hyperplane extraction. "
                           f"Supported types: LinearSVC, LogisticRegression, Perceptron, "
                           f"LinearRegression, Ridge, Lasso, LinearDiscriminantAnalysis, GaussianNB")

    def get_hyperplane(self) -> Hyperplane:
        """
        Get the extracted hyperplane object.
        
        Returns:
        --------
        Hyperplane
            Hyperplane object for visualization and analysis
        """
        if self.is_1d:
            raise ValueError("1D models don't have hyperplane objects. Use plot_linear_regression() instead.")
        return self.hyperplane

    def equation(self) -> str:
        """
        Get the hyperplane equation.
        
        Returns:
        --------
        str
            Mathematical equation
        """
        if self.is_1d:
            return f"y = {self.w[0]:.3f}*x + {self.b:.3f}"
        return self.hyperplane.equation()

    def plot_2d(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
                show_margin: bool = False, **kwargs) -> None:
        """
        Plot 2D hyperplane with model-specific enhancements.
        
        Parameters:
        -----------
        X : array-like, optional
            Training data
        y : array-like, optional
            Training labels
        show_margin : bool, default False
            Show margin for SVM-like models
        **kwargs
            Additional plotting parameters
        """
        if self.is_1d:
            self.plot_linear_regression(X, y, **kwargs)
            return
            
        # Enable margin for SVM models
        if self.model_type in ["SVC", "LinearSVC"] and show_margin:
            kwargs['show_margin'] = True
            
        self.hyperplane.plot_2d(X=X, y=y, **kwargs)

    def plot_3d(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None, 
                **kwargs) -> None:
        """
        Plot 3D hyperplane.
        
        Parameters:
        -----------
        X : array-like, optional
            Training data
        y : array-like, optional
            Training labels
        **kwargs
            Additional plotting parameters
        """
        if self.is_1d:
            raise ValueError("1D models cannot be plotted in 3D")
        self.hyperplane.plot_3d(X=X, y=y, **kwargs)

    def plot_decision_regions(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        """
        Plot decision regions.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
        **kwargs
            Additional plotting parameters
        """
        if self.is_1d:
            raise ValueError("Decision regions not applicable for 1D regression models")
        self.hyperplane.plot_decision_regions(X, y, **kwargs)

    # Algorithm-specific plotting methods
    def plot_svm(self, X: np.ndarray, y: np.ndarray, show_support_vectors: bool = True,
                 margin_style: str = 'dashed', figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        SVM-specific hyperplane plotting with support vectors and margins.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
        show_support_vectors : bool, default True
            Highlight support vectors
        margin_style : str, default 'dashed'
            Style for margin lines ('dashed', 'dotted', 'solid')
        figsize : tuple, default (12, 8)
            Figure size
        """
        if self.model_type not in ['SVC', 'LinearSVC']:
            warnings.warn(f"plot_svm() called on {self.model_type}, not an SVM model")
        
        plt.figure(figsize=figsize)
        
        # Plot decision regions first
        if len(self.w) == 2:
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                                np.linspace(y_min, y_max, 100))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = self.hyperplane.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            plt.contourf(xx, yy, Z, alpha=0.2, cmap='RdYlBu')
        
        # Plot data points
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       alpha=0.7, s=60, 
                       label=f'Class {cls}',
                       edgecolors='black')
        
        # Plot decision boundary
        if len(self.w) == 2:
            w1, w2 = self.w
            b = self.b
            x_range = [x_min, x_max]
            
            if abs(w2) > 1e-10:
                xs = np.linspace(*x_range, 100)
                ys = -(w1 * xs + b) / w2
                plt.plot(xs, ys, 'k-', linewidth=3, label="Decision Boundary")
                
                # Plot margins
                margin = 1.0 / np.linalg.norm(self.w)
                margin_b_pos = b - np.linalg.norm(self.w) * margin
                margin_b_neg = b + np.linalg.norm(self.w) * margin
                
                ys_margin_pos = -(w1 * xs + margin_b_pos) / w2
                ys_margin_neg = -(w1 * xs + margin_b_neg) / w2
                
                linestyle = '--' if margin_style == 'dashed' else (':' if margin_style == 'dotted' else '-')
                plt.plot(xs, ys_margin_pos, 'k' + linestyle, alpha=0.7, label="Margin +")
                plt.plot(xs, ys_margin_neg, 'k' + linestyle, alpha=0.7, label="Margin -")
        
        # Highlight support vectors if available and requested
        if show_support_vectors and hasattr(self.model, 'support_'):
            support_vectors = X[self.model.support_]
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                       s=200, facecolors='none', edgecolors='black', 
                       linewidths=2, label='Support Vectors')
        
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title(f"SVM Hyperplane: {self.equation()}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_logistic_regression(self, X: np.ndarray, y: np.ndarray, 
                                show_probabilities: bool = True,
                                probability_contours: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9],
                                figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Logistic Regression-specific plotting with probability contours.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
        show_probabilities : bool, default True
            Show probability contours
        probability_contours : list, default [0.1, 0.3, 0.5, 0.7, 0.9]
            Probability levels to show as contours
        figsize : tuple, default (12, 8)
            Figure size
        """
        if self.model_type != 'LogisticRegression':
            warnings.warn(f"plot_logistic_regression() called on {self.model_type}")
        
        if len(self.w) != 2:
            raise ValueError("Logistic regression plotting only supports 2D data")
        
        plt.figure(figsize=figsize)
        
        # Create mesh for probability calculation
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Calculate probabilities if model supports it
        if hasattr(self.model, 'predict_proba') and show_probabilities:
            Z_proba = self.model.predict_proba(mesh_points)[:, 1]
            Z_proba = Z_proba.reshape(xx.shape)
            
            # Plot probability contours
            contours = plt.contour(xx, yy, Z_proba, levels=probability_contours, 
                                  colors='gray', alpha=0.6, linestyles='dashed')
            plt.clabel(contours, inline=True, fontsize=10, fmt='%.1f')
            
            # Fill regions
            plt.contourf(xx, yy, Z_proba, levels=50, alpha=0.3, cmap='RdYlBu')
        
        # Plot decision boundary (probability = 0.5)
        w1, w2 = self.w
        b = self.b
        
        if abs(w2) > 1e-10:
            xs = np.linspace(x_min, x_max, 100)
            ys = -(w1 * xs + b) / w2
            plt.plot(xs, ys, 'k-', linewidth=3, label="Decision Boundary (P=0.5)")
        
        # Plot data points
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], 
                       alpha=0.8, s=60, 
                       label=f'Class {cls}',
                       edgecolors='black')
        
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title(f"Logistic Regression: {self.equation()}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_perceptron(self, X: np.ndarray, y: np.ndarray, 
                       show_misclassified: bool = True,
                       figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Perceptron-specific plotting highlighting misclassified points.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
        show_misclassified : bool, default True
            Highlight misclassified points
        figsize : tuple, default (10, 6)
            Figure size
        """
        if self.model_type != 'Perceptron':
            warnings.warn(f"plot_perceptron() called on {self.model_type}")
        
        if len(self.w) != 2:
            raise ValueError("Perceptron plotting only supports 2D data")
        
        plt.figure(figsize=figsize)
        
        # Plot decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.hyperplane.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # Get predictions for highlighting misclassified points
        y_pred = self.model.predict(X)
        correct_mask = y == y_pred
        incorrect_mask = y != y_pred
        
        # Plot correctly classified points
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple']
        
        for i, cls in enumerate(unique_classes):
            cls_mask = y == cls
            correct_cls_mask = cls_mask & correct_mask
            incorrect_cls_mask = cls_mask & incorrect_mask
            
            # Correctly classified points
            if np.any(correct_cls_mask):
                plt.scatter(X[correct_cls_mask, 0], X[correct_cls_mask, 1], 
                           c=colors[i % len(colors)], alpha=0.8, s=60, 
                           label=f'Class {cls} (Correct)',
                           edgecolors='black', marker='o')
            
            # Misclassified points
            if show_misclassified and np.any(incorrect_cls_mask):
                plt.scatter(X[incorrect_cls_mask, 0], X[incorrect_cls_mask, 1], 
                           c=colors[i % len(colors)], alpha=0.8, s=100, 
                           label=f'Class {cls} (Misclassified)',
                           edgecolors='red', linewidths=3, marker='X')
        
        # Plot decision boundary
        w1, w2 = self.w
        b = self.b
        
        if abs(w2) > 1e-10:
            xs = np.linspace(x_min, x_max, 100)
            ys = -(w1 * xs + b) / w2
            plt.plot(xs, ys, 'k-', linewidth=3, label="Decision Boundary")
        
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title(f"Perceptron: {self.equation()}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_lda(self, X: np.ndarray, y: np.ndarray, 
                 show_class_centers: bool = True,
                 show_projections: bool = False,
                 figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        LDA-specific plotting with class centers and projections.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
        show_class_centers : bool, default True
            Show class center points
        show_projections : bool, default False
            Show projected points onto discriminant direction
        figsize : tuple, default (10, 6)
            Figure size
        """
        if self.model_type != 'LinearDiscriminantAnalysis':
            warnings.warn(f"plot_lda() called on {self.model_type}")
        
        if len(self.w) != 2:
            raise ValueError("LDA plotting only supports 2D data")
        
        plt.figure(figsize=figsize)
        
        # Plot decision regions
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.hyperplane.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
        
        # Plot data points
        unique_classes = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple']
        class_centers = []
        
        for i, cls in enumerate(unique_classes):
            mask = y == cls
            plt.scatter(X[mask, 0], X[mask, 1], 
                       c=colors[i % len(colors)], alpha=0.7, s=60, 
                       label=f'Class {cls}',
                       edgecolors='black')
            
            if show_class_centers:
                center = np.mean(X[mask], axis=0)
                class_centers.append(center)
                plt.scatter(center[0], center[1], 
                           c=colors[i % len(colors)], s=200, 
                           marker='*', edgecolors='black', linewidths=2,
                           label=f'Class {cls} Center')
        
        # Plot decision boundary
        w1, w2 = self.w
        b = self.b
        
        if abs(w2) > 1e-10:
            xs = np.linspace(x_min, x_max, 100)
            ys = -(w1 * xs + b) / w2
            plt.plot(xs, ys, 'k-', linewidth=3, label="LDA Decision Boundary")
        
        # Show projections if requested
        if show_projections and len(class_centers) == 2:
            # Draw line connecting class centers
            center1, center2 = class_centers
            plt.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                    'g--', linewidth=2, alpha=0.7, label='Between-class direction')
        
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        plt.title(f"Linear Discriminant Analysis: {self.equation()}", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_linear_regression(self, X: np.ndarray, y: np.ndarray, 
                              show_residuals: bool = True,
                              confidence_interval: bool = False,
                              figsize: Tuple[int, int] = (10, 6)) -> None:
        """
        Linear Regression-specific plotting with residuals.
        
        Parameters:
        -----------
        X : array-like
            Training data (1D or 2D)
        y : array-like
            Target values
        show_residuals : bool, default True
            Show residual lines
        confidence_interval : bool, default False
            Show confidence interval (for 1D case)
        figsize : tuple, default (10, 6)
            Figure size
        """
        if 'Regression' not in self.model_type and 'Ridge' not in self.model_type and 'Lasso' not in self.model_type:
            warnings.warn(f"plot_linear_regression() called on {self.model_type}")
        
        # Handle 1D case (simple linear regression)
        if X.shape[1] == 1:
            plt.figure(figsize=figsize)
            
            # Plot data points
            plt.scatter(X[:, 0], y, alpha=0.7, s=60, color='blue', label='Data Points')
            
            # Plot regression line
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
            y_pred_line = self.w[0] * x_range + self.b
            plt.plot(x_range, y_pred_line, 'r-', linewidth=2, label='Regression Line')
            
            # Show residuals
            if show_residuals:
                y_pred = self.model.predict(X)
                for i in range(len(X)):
                    plt.plot([X[i, 0], X[i, 0]], [y[i], y_pred[i]], 
                            'k--', alpha=0.5, linewidth=1)
            
            plt.xlabel("Feature", fontsize=12)
            plt.ylabel("Target", fontsize=12)
            plt.title(f"Linear Regression: y = {self.w[0]:.3f}*x + {self.b:.3f}", fontsize=14)
            
        # Handle 2D case (multiple regression - show as surface)
        elif X.shape[1] == 2:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot data points
            ax.scatter(X[:, 0], X[:, 1], y, alpha=0.7, s=60, color='blue', label='Data Points')
            
            # Plot regression plane
            x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 20)
            y_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 20)
            xx, yy = np.meshgrid(x_range, y_range)
            zz = self.w[0] * xx + self.w[1] * yy + self.b
            
            ax.plot_surface(xx, yy, zz, alpha=0.3, color='red')
            
            # Show residuals
            if show_residuals:
                y_pred = self.model.predict(X)
                for i in range(len(X)):
                    ax.plot([X[i, 0], X[i, 0]], [X[i, 1], X[i, 1]], 
                           [y[i], y_pred[i]], 'k--', alpha=0.5, linewidth=1)
            
            ax.set_xlabel("Feature 1", fontsize=12)
            ax.set_ylabel("Feature 2", fontsize=12)
            ax.set_zlabel("Target", fontsize=12)
            ax.set_title(f"Linear Regression: {self.equation()}", fontsize=14)
        
        else:
            raise ValueError("Linear regression plotting supports only 1D or 2D features")
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_algorithm_comparison(self, models_dict: dict, X: np.ndarray, y: np.ndarray,
                                 figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Compare multiple algorithm hyperplanes in subplots.
        
        Parameters:
        -----------
        models_dict : dict
            Dictionary of {algorithm_name: trained_model}
        X : array-like
            Training data
        y : array-like
            Training labels
        figsize : tuple, default (15, 10)
            Figure size
        """
        n_models = len(models_dict)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (name, model) in enumerate(models_dict.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Extract hyperplane
            extractor = HyperplaneExtractor(model)
            
            # Plot decision regions
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50),
                                np.linspace(y_min, y_max, 50))
            
            mesh_points = np.c_[xx.ravel(), yy.ravel()]
            Z = extractor.hyperplane.predict(mesh_points)
            Z = Z.reshape(xx.shape)
            
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
            
            # Plot data points
            unique_classes = np.unique(y)
            colors = ['red', 'blue', 'green', 'purple']
            
            for i, cls in enumerate(unique_classes):
                mask = y == cls
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=colors[i % len(colors)], alpha=0.7, s=40,
                          edgecolors='black', linewidth=0.5)
            
            # Plot decision boundary
            w1, w2 = extractor.w
            b = extractor.b
            
            if abs(w2) > 1e-10:
                xs = np.linspace(x_min, x_max, 100)
                ys = -(w1 * xs + b) / w2
                ax.plot(xs, ys, 'k-', linewidth=2)
            
            ax.set_xlabel("Feature 1", fontsize=10)
            ax.set_ylabel("Feature 2", fontsize=10)
            ax.set_title(f"{name}\n{extractor.equation()}", fontsize=11)
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_models, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            ax.set_visible(False)
        
        plt.tight_layout()
        plt.show()

    def analyze_model(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Comprehensive analysis of the model's hyperplane.
        
        Parameters:
        -----------
        X : array-like
            Training data
        y : array-like
            Training labels
            
        Returns:
        --------
        dict
            Analysis results
        """
        X = np.array(X)
        y = np.array(y)
        
        # Basic hyperplane info
        analysis = {
            'model_type': self.model_type,
            'equation': self.equation(),
            'weights': self.w,
            'bias': self.b,
            'weight_magnitude': np.linalg.norm(self.w),
        }
        
        # Calculate distances for all points
        distances = []
        for point in X:
            distances.append(self.hyperplane.distance(point))
        distances = np.array(distances)
        
        analysis.update({
            'mean_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'std_distance': np.std(distances)
        })
        
        # Margin analysis for binary classification
        if len(np.unique(y)) == 2:
            classes = np.unique(y)
            class_distances = {}
            
            for cls in classes:
                mask = y == cls
                cls_distances = distances[mask]
                class_distances[f'class_{cls}_mean_distance'] = np.mean(cls_distances)
                class_distances[f'class_{cls}_min_distance'] = np.min(cls_distances)
            
            analysis.update(class_distances)
            analysis['margin'] = 2 / np.linalg.norm(self.w)  # SVM-style margin
        
        return analysis

    def compare_models(self, other_extractor: 'HyperplaneExtractor') -> dict:
        """
        Compare two hyperplane models.
        
        Parameters:
        -----------
        other_extractor : HyperplaneExtractor
            Another hyperplane extractor to compare with
            
        Returns:
        --------
        dict
            Comparison results
        """
        # Weight comparison
        weight_diff = np.linalg.norm(self.w - other_extractor.w)
        bias_diff = abs(self.b - other_extractor.b)
        
        # Angle between normal vectors
        cos_angle = np.dot(self.w, other_extractor.w) / (
            np.linalg.norm(self.w) * np.linalg.norm(other_extractor.w)
        )
        angle_degrees = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
        
        return {
            'model1_type': self.model_type,
            'model2_type': other_extractor.model_type,
            'weight_difference': weight_diff,
            'bias_difference': bias_diff,
            'angle_between_normals': angle_degrees,
            'weight_magnitude_ratio': np.linalg.norm(self.w) / np.linalg.norm(other_extractor.w)
        }


# Convenience functions for quick hyperplane creation
def create_hyperplane_from_points(points: List[np.ndarray]) -> Hyperplane:
    """
    Create hyperplane from points (2D: 2 points, 3D: 3 points).
    
    Parameters:
    -----------
    points : list of arrays
        Points that define the hyperplane
        
    Returns:
    --------
    Hyperplane
        Fitted hyperplane
    """
    points = [np.array(p) for p in points]
    
    if len(points[0]) == 2:  # 2D case
        if len(points) != 2:
            raise ValueError("Need exactly 2 points for 2D hyperplane")
        
        p1, p2 = points
        # Line equation from two points
        direction = p2 - p1
        normal = np.array([-direction[1], direction[0]])  # Perpendicular
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # ax + by + c = 0, where (a,b) is normal and c = -(a*x0 + b*y0)
        bias = -np.dot(normal, p1)
        
        return Hyperplane(normal, bias)
    
    elif len(points[0]) == 3:  # 3D case
        if len(points) != 3:
            raise ValueError("Need exactly 3 points for 3D hyperplane")
        
        p1, p2, p3 = points
        # Plane equation from three points
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)  # Normalize
        
        # ax + by + cz + d = 0
        bias = -np.dot(normal, p1)
        
        return Hyperplane(normal, bias)
    
    else:
        raise ValueError("Only 2D and 3D hyperplanes supported")


def extract_hyperplane(model) -> HyperplaneExtractor:
    """
    Convenience function to extract hyperplane from any supported model.
    
    Parameters:
    -----------
    model : sklearn model
        Trained ML model
        
    Returns:
    --------
    HyperplaneExtractor
        Hyperplane extractor object
    """
    return HyperplaneExtractor(model)


# Convenience functions for algorithm-specific plotting
def plot_svm_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Quick SVM hyperplane plotting with margins and support vectors.
    
    Parameters:
    -----------
    model : sklearn SVM model
        Trained SVM model
    X : array-like
        Training data
    y : array-like
        Training labels
    **kwargs
        Additional plotting parameters
    """
    extractor = HyperplaneExtractor(model)
    extractor.plot_svm(X, y, **kwargs)


def plot_logistic_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Quick Logistic Regression hyperplane plotting with probability contours.
    
    Parameters:
    -----------
    model : sklearn LogisticRegression model
        Trained logistic regression model
    X : array-like
        Training data
    y : array-like
        Training labels
    **kwargs
        Additional plotting parameters
    """
    extractor = HyperplaneExtractor(model)
    extractor.plot_logistic_regression(X, y, **kwargs)


def plot_perceptron_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Quick Perceptron hyperplane plotting with misclassified points highlighted.
    
    Parameters:
    -----------
    model : sklearn Perceptron model
        Trained perceptron model
    X : array-like
        Training data
    y : array-like
        Training labels
    **kwargs
        Additional plotting parameters
    """
    extractor = HyperplaneExtractor(model)
    extractor.plot_perceptron(X, y, **kwargs)


def plot_lda_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Quick LDA hyperplane plotting with class centers.
    
    Parameters:
    -----------
    model : sklearn LDA model
        Trained LDA model
    X : array-like
        Training data
    y : array-like
        Training labels
    **kwargs
        Additional plotting parameters
    """
    extractor = HyperplaneExtractor(model)
    extractor.plot_lda(X, y, **kwargs)


def plot_linear_regression_hyperplane(model, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Quick Linear Regression hyperplane plotting with residuals.
    
    Parameters:
    -----------
    model : sklearn regression model
        Trained regression model
    X : array-like
        Training data
    y : array-like
        Target values
    **kwargs
        Additional plotting parameters
    """
    extractor = HyperplaneExtractor(model)
    extractor.plot_linear_regression(X, y, **kwargs)


def compare_algorithm_hyperplanes(models_dict: dict, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
    """
    Compare multiple algorithm hyperplanes in a single visualization.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {algorithm_name: trained_model}
    X : array-like
        Training data
    y : array-like
        Training labels
    **kwargs
        Additional plotting parameters
    """
    # Use the first model to create an extractor for the comparison method
    first_model = list(models_dict.values())[0]
    extractor = HyperplaneExtractor(first_model)
    extractor.plot_algorithm_comparison(models_dict, X, y, **kwargs)