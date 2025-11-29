try:
    import shap
except ImportError:
    shap = None
import matplotlib.pyplot as plt
import pandas as pd

def explain_shap(model, X):
    """
    Provides SHAP-based explainability for ML models.
    """
    if shap is None:
        print("SHAP library not installed. Please install it using 'pip install shap'")
        return

    # SHAP explainer depends on the model type
    try:
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)
        
        plt.figure()
        shap.summary_plot(shap_values, X)
        plt.show()
        
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        print("Trying KernelExplainer (slower)...")
        try:
            # Fallback for models not supported by TreeExplainer/LinearExplainer
            # Using a sample of X for speed if X is large
            X_sample = shap.sample(X, 100) if len(X) > 100 else X
            explainer = shap.KernelExplainer(model.predict, X_sample)
            shap_values = explainer.shap_values(X_sample)
            
            plt.figure()
            shap.summary_plot(shap_values, X_sample)
            plt.show()
        except Exception as e2:
            print(f"Could not generate SHAP explanation: {e2}")
