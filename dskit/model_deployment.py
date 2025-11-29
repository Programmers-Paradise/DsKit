import pandas as pd
import numpy as np
import pickle
import joblib
import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

class ModelPersistence:
    """
    Advanced model persistence and serialization utilities.
    """
    
    def __init__(self):
        self.supported_formats = ['pickle', 'joblib', 'onnx', 'pmml']
        self.model_registry = {}
    
    def save_model(self, model, filepath: str, format: str = 'auto', 
                   metadata: dict = None, compress: bool = True) -> bool:
        """Save model with metadata and compression options."""
        
        if format == 'auto':
            # Infer format from extension
            extension = Path(filepath).suffix.lower()
            format_mapping = {
                '.pkl': 'pickle',
                '.pickle': 'pickle',
                '.joblib': 'joblib',
                '.onnx': 'onnx',
                '.pmml': 'pmml'
            }
            format = format_mapping.get(extension, 'pickle')
        
        # Prepare model package
        model_package = {
            'model': model,
            'metadata': metadata or {},
            'format_version': '1.0',
            'saved_timestamp': pd.Timestamp.now().isoformat()
        }
        
        # Add model type information
        model_package['metadata']['model_type'] = type(model).__name__
        model_package['metadata']['model_module'] = type(model).__module__
        
        try:
            if format == 'pickle':
                compression = 'gzip' if compress else None
                with open(filepath, 'wb') as f:
                    pickle.dump(model_package, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            elif format == 'joblib':
                compression = 'gzip' if compress else None
                joblib.dump(model_package, filepath, compress=compression)
            
            elif format == 'onnx':
                # ONNX export (simplified - would need sklearn-onnx)
                print("ONNX export would require sklearn-onnx package")
                return False
            
            elif format == 'pmml':
                # PMML export (simplified - would need sklearn2pmml)
                print("PMML export would require sklearn2pmml package")
                return False
            
            print(f"✅ Model saved to {filepath} ({format} format)")
            
            # Register in local registry
            model_id = Path(filepath).stem
            self.model_registry[model_id] = {
                'filepath': filepath,
                'format': format,
                'metadata': model_package['metadata'],
                'saved_at': model_package['saved_timestamp']
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Model save failed: {e}")
            return False
    
    def load_model(self, filepath: str, format: str = 'auto'):
        """Load model with automatic format detection."""
        
        if format == 'auto':
            extension = Path(filepath).suffix.lower()
            format_mapping = {
                '.pkl': 'pickle',
                '.pickle': 'pickle',
                '.joblib': 'joblib'
            }
            format = format_mapping.get(extension, 'pickle')
        
        try:
            if format == 'pickle':
                with open(filepath, 'rb') as f:
                    model_package = pickle.load(f)
            
            elif format == 'joblib':
                model_package = joblib.load(filepath)
            
            else:
                print(f"Unsupported format for loading: {format}")
                return None
            
            # Extract model and metadata
            model = model_package.get('model')
            metadata = model_package.get('metadata', {})
            
            print(f"✅ Model loaded from {filepath}")
            print(f"📋 Model type: {metadata.get('model_type', 'Unknown')}")
            print(f"📅 Saved: {model_package.get('saved_timestamp', 'Unknown')}")
            
            return model, metadata
            
        except Exception as e:
            print(f"❌ Model load failed: {e}")
            return None, {}
    
    def save_pipeline(self, pipeline, pipeline_name: str, base_path: str = "./models") -> bool:
        """Save complete ML pipeline with all components."""
        
        base_path = Path(base_path)
        base_path.mkdir(exist_ok=True)
        
        pipeline_dir = base_path / pipeline_name
        pipeline_dir.mkdir(exist_ok=True)
        
        try:
            # Save main pipeline
            main_path = pipeline_dir / "pipeline.joblib"
            self.save_model(pipeline, str(main_path), format='joblib')
            
            # Save individual components if it's a sklearn pipeline
            if hasattr(pipeline, 'steps'):
                components_dir = pipeline_dir / "components"
                components_dir.mkdir(exist_ok=True)
                
                for step_name, component in pipeline.steps:
                    component_path = components_dir / f"{step_name}.joblib"
                    self.save_model(component, str(component_path), format='joblib')
            
            # Create manifest file
            manifest = {
                'pipeline_name': pipeline_name,
                'created_at': pd.Timestamp.now().isoformat(),
                'pipeline_type': type(pipeline).__name__,
                'components': []
            }
            
            if hasattr(pipeline, 'steps'):
                manifest['components'] = [
                    {
                        'name': step_name,
                        'type': type(component).__name__,
                        'module': type(component).__module__
                    }
                    for step_name, component in pipeline.steps
                ]
            
            manifest_path = pipeline_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            
            print(f"✅ Pipeline '{pipeline_name}' saved to {pipeline_dir}")
            return True
            
        except Exception as e:
            print(f"❌ Pipeline save failed: {e}")
            return False
    
    def load_pipeline(self, pipeline_name: str, base_path: str = "./models"):
        """Load complete ML pipeline."""
        
        pipeline_dir = Path(base_path) / pipeline_name
        
        if not pipeline_dir.exists():
            print(f"❌ Pipeline directory not found: {pipeline_dir}")
            return None
        
        try:
            # Load manifest
            manifest_path = pipeline_dir / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                print(f"📋 Loading pipeline: {manifest.get('pipeline_name')}")
                print(f"📅 Created: {manifest.get('created_at')}")
            
            # Load main pipeline
            main_path = pipeline_dir / "pipeline.joblib"
            pipeline, metadata = self.load_model(str(main_path), format='joblib')
            
            return pipeline
            
        except Exception as e:
            print(f"❌ Pipeline load failed: {e}")
            return None
    
    def list_saved_models(self, base_path: str = "./models") -> pd.DataFrame:
        """List all saved models and pipelines."""
        
        base_path = Path(base_path)
        if not base_path.exists():
            return pd.DataFrame()
        
        models_info = []
        
        # Search for model files
        for file_path in base_path.rglob("*.joblib"):
            try:
                # Try to load metadata
                model_package = joblib.load(file_path)
                if isinstance(model_package, dict) and 'metadata' in model_package:
                    metadata = model_package['metadata']
                    models_info.append({
                        'name': file_path.stem,
                        'path': str(file_path),
                        'type': metadata.get('model_type', 'Unknown'),
                        'saved_at': model_package.get('saved_timestamp', 'Unknown'),
                        'size_mb': file_path.stat().st_size / (1024**2)
                    })
            except Exception:
                # If loading fails, just record basic info
                models_info.append({
                    'name': file_path.stem,
                    'path': str(file_path),
                    'type': 'Unknown',
                    'saved_at': 'Unknown',
                    'size_mb': file_path.stat().st_size / (1024**2)
                })
        
        return pd.DataFrame(models_info)

class ConfigManager:
    """
    Advanced configuration management for ML projects.
    """
    
    def __init__(self, config_path: str = "./config.json"):
        self.config_path = Path(config_path)
        self.config = {}
        self.load_config()
    
    def load_config(self):
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"✅ Configuration loaded from {self.config_path}")
            except Exception as e:
                print(f"❌ Failed to load config: {e}")
                self.config = {}
        else:
            self.config = self._get_default_config()
            self.save_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"❌ Failed to save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value
    
    def set(self, key: str, value):
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
        self.save_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'data': {
                'default_path': './data',
                'cache_enabled': True,
                'sample_size': 10000
            },
            'models': {
                'default_path': './models',
                'auto_save': True,
                'compression': True
            },
            'preprocessing': {
                'handle_missing': 'auto',
                'scale_features': True,
                'encode_categorical': 'auto'
            },
            'visualization': {
                'default_backend': 'matplotlib',
                'figure_size': [10, 6],
                'color_palette': 'viridis'
            },
            'logging': {
                'level': 'INFO',
                'log_to_file': False,
                'log_path': './logs'
            }
        }

class ExperimentTracker:
    """
    Simple experiment tracking and versioning.
    """
    
    def __init__(self, experiments_path: str = "./experiments"):
        self.experiments_path = Path(experiments_path)
        self.experiments_path.mkdir(exist_ok=True)
        self.current_experiment = None
    
    def start_experiment(self, experiment_name: str, description: str = "") -> str:
        """Start a new experiment."""
        
        # Create unique experiment ID
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{experiment_name}_{timestamp}"
        
        experiment_dir = self.experiments_path / experiment_id
        experiment_dir.mkdir(exist_ok=True)
        
        # Create experiment metadata
        experiment_metadata = {
            'id': experiment_id,
            'name': experiment_name,
            'description': description,
            'started_at': pd.Timestamp.now().isoformat(),
            'status': 'running',
            'metrics': {},
            'parameters': {},
            'artifacts': []
        }
        
        # Save metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        
        self.current_experiment = experiment_id
        print(f"🚀 Started experiment: {experiment_id}")
        
        return experiment_id
    
    def log_parameter(self, key: str, value):
        """Log a parameter for the current experiment."""
        if not self.current_experiment:
            print("❌ No active experiment. Start an experiment first.")
            return
        
        experiment_dir = self.experiments_path / self.current_experiment
        metadata_path = experiment_dir / "metadata.json"
        
        # Load current metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update parameters
        metadata['parameters'][key] = value
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📝 Logged parameter: {key} = {value}")
    
    def log_metric(self, key: str, value: float, step: int = None):
        """Log a metric for the current experiment."""
        if not self.current_experiment:
            print("❌ No active experiment. Start an experiment first.")
            return
        
        experiment_dir = self.experiments_path / self.current_experiment
        metadata_path = experiment_dir / "metadata.json"
        
        # Load current metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update metrics
        if key not in metadata['metrics']:
            metadata['metrics'][key] = []
        
        metric_entry = {
            'value': float(value),
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        if step is not None:
            metric_entry['step'] = step
        
        metadata['metrics'][key].append(metric_entry)
        
        # Save updated metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📊 Logged metric: {key} = {value}")
    
    def log_artifact(self, artifact_path: str, artifact_type: str = "file"):
        """Log an artifact for the current experiment."""
        if not self.current_experiment:
            print("❌ No active experiment. Start an experiment first.")
            return
        
        experiment_dir = self.experiments_path / self.current_experiment
        artifacts_dir = experiment_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        # Copy artifact to experiment directory
        artifact_src = Path(artifact_path)
        if artifact_src.exists():
            artifact_dst = artifacts_dir / artifact_src.name
            
            if artifact_src.is_file():
                import shutil
                shutil.copy2(artifact_src, artifact_dst)
            else:
                import shutil
                shutil.copytree(artifact_src, artifact_dst, dirs_exist_ok=True)
        
        # Update metadata
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        artifact_info = {
            'name': artifact_src.name,
            'type': artifact_type,
            'original_path': str(artifact_path),
            'experiment_path': str(artifact_dst),
            'logged_at': pd.Timestamp.now().isoformat()
        }
        
        metadata['artifacts'].append(artifact_info)
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"📁 Logged artifact: {artifact_src.name}")
    
    def end_experiment(self, status: str = "completed"):
        """End the current experiment."""
        if not self.current_experiment:
            print("❌ No active experiment.")
            return
        
        experiment_dir = self.experiments_path / self.current_experiment
        metadata_path = experiment_dir / "metadata.json"
        
        # Update metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata['status'] = status
        metadata['ended_at'] = pd.Timestamp.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"🏁 Ended experiment: {self.current_experiment} (Status: {status})")
        self.current_experiment = None
    
    def list_experiments(self) -> pd.DataFrame:
        """List all experiments."""
        experiments = []
        
        for exp_dir in self.experiments_path.iterdir():
            if exp_dir.is_dir():
                metadata_path = exp_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Get latest metrics
                        latest_metrics = {}
                        for metric_name, metric_values in metadata.get('metrics', {}).items():
                            if metric_values:
                                latest_metrics[metric_name] = metric_values[-1]['value']
                        
                        experiments.append({
                            'id': metadata.get('id'),
                            'name': metadata.get('name'),
                            'status': metadata.get('status'),
                            'started_at': metadata.get('started_at'),
                            'ended_at': metadata.get('ended_at'),
                            'parameters_count': len(metadata.get('parameters', {})),
                            'metrics_count': len(metadata.get('metrics', {})),
                            'artifacts_count': len(metadata.get('artifacts', [])),
                            **latest_metrics
                        })
                    except Exception as e:
                        print(f"Error loading experiment {exp_dir.name}: {e}")
        
        return pd.DataFrame(experiments)

class DataVersioning:
    """
    Simple data versioning and lineage tracking.
    """
    
    def __init__(self, versions_path: str = "./data_versions"):
        self.versions_path = Path(versions_path)
        self.versions_path.mkdir(exist_ok=True)
    
    def create_version(self, df: pd.DataFrame, dataset_name: str, 
                      description: str = "", parent_version: str = None) -> str:
        """Create a new version of a dataset."""
        
        # Generate version ID
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{dataset_name}_v{timestamp}"
        
        version_dir = self.versions_path / version_id
        version_dir.mkdir(exist_ok=True)
        
        # Save data
        data_path = version_dir / "data.parquet"
        df.to_parquet(data_path, index=False)
        
        # Create version metadata
        version_metadata = {
            'version_id': version_id,
            'dataset_name': dataset_name,
            'description': description,
            'created_at': pd.Timestamp.now().isoformat(),
            'parent_version': parent_version,
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
            'hash': hash(str(df.values.tobytes()))  # Simple hash
        }
        
        # Save metadata
        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(version_metadata, f, indent=2, default=str)
        
        print(f"📦 Created data version: {version_id}")
        return version_id
    
    def load_version(self, version_id: str) -> pd.DataFrame:
        """Load a specific version of a dataset."""
        
        version_dir = self.versions_path / version_id
        data_path = version_dir / "data.parquet"
        
        if not data_path.exists():
            print(f"❌ Version not found: {version_id}")
            return pd.DataFrame()
        
        try:
            df = pd.read_parquet(data_path)
            print(f"✅ Loaded data version: {version_id}")
            print(f"📊 Shape: {df.shape}")
            return df
        except Exception as e:
            print(f"❌ Failed to load version: {e}")
            return pd.DataFrame()
    
    def list_versions(self, dataset_name: str = None) -> pd.DataFrame:
        """List all versions of datasets."""
        
        versions = []
        
        for version_dir in self.versions_path.iterdir():
            if version_dir.is_dir():
                metadata_path = version_dir / "metadata.json"
                if metadata_path.exists():
                    try:
                        with open(metadata_path, 'r') as f:
                            metadata = json.load(f)
                        
                        # Filter by dataset name if specified
                        if dataset_name is None or metadata.get('dataset_name') == dataset_name:
                            versions.append(metadata)
                    except Exception as e:
                        print(f"Error loading version metadata from {version_dir.name}: {e}")
        
        return pd.DataFrame(versions)