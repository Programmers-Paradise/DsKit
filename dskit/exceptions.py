class DskitError(Exception):
    """Base exception for all errors."""
    pass


class ModelNotFoundError(DskitError):
    """Raised when a requested model type is not available."""
    
    def __init__(self, model_name, available_models):
        self.model_name = model_name
        self.available_models = list(available_models)
        message = f"Model '{model_name}' not supported. Available: {', '.join(self.available_models)}"
        super().__init__(message)


class ModelNotTrainedError(DskitError):
    """Raised when attempting to use a model that hasn't been trained."""
    
    def __init__(self, operation="predict"):
        message = f"Cannot {operation}: model not trained yet. Call fit() first."
        super().__init__(message)


class InvalidParameterError(DskitError):
    """Raised when a parameter has an invalid value."""
    
    def __init__(self, param_name, value, valid_values=None):
        self.param_name = param_name
        self.value = value
        if valid_values:
            message = f"Invalid '{param_name}': '{value}'. Valid: {valid_values}"
        else:
            message = f"Invalid value '{value}' for parameter '{param_name}'"
        super().__init__(message)


class DependencyError(DskitError):
    """Raised when an optional dependency is not available."""
    
    def __init__(self, package_name, feature, install_cmd=None):
        self.package_name = package_name
        install_cmd = install_cmd or f"pip install {package_name}"
        message = f"'{package_name}' required for {feature}. Install: {install_cmd}"
        super().__init__(message)


class DataValidationError(DskitError):
    """Raised when input data fails validation."""
    pass


class ColumnNotFoundError(DataValidationError):
    """Raised when a required column is not found."""
    
    def __init__(self, column_name, available_columns):
        self.column_name = column_name
        message = f"Column '{column_name}' not found. Available: {list(available_columns)[:10]}"
        super().__init__(message)
