"""
Model Registry - Universal Business Solution Framework

Model versioning, storage, and lifecycle management.
Track experiments, compare models, and manage deployments.

Usage:
```python
from patterns.ml import ModelRegistry, ModelVersion

# Initialize registry
registry = ModelRegistry(storage_path='models/')

# Register a model
version = registry.register(
    name='price_predictor',
    model=trained_model,
    metrics={'rmse': 0.05, 'mae': 0.03},
    tags=['production', 'v2']
)

# Load model for inference
model = registry.load('price_predictor', version='latest')

# Compare model versions
comparison = registry.compare_versions('price_predictor')
```
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime
import os


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class SerializationFormat(Enum):
    """Model serialization formats"""
    PICKLE = "pickle"
    JSON = "json"
    JOBLIB = "joblib"


@dataclass
class ModelMetadata:
    """Metadata for a model version"""
    name: str
    version: str
    created_at: datetime
    stage: ModelStage
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    tags: List[str]
    description: str
    artifact_path: str
    checksum: str
    framework: str = "unknown"
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'stage': self.stage.value,
            'metrics': self.metrics,
            'parameters': self.parameters,
            'tags': self.tags,
            'description': self.description,
            'artifact_path': self.artifact_path,
            'checksum': self.checksum,
            'framework': self.framework,
            'input_schema': self.input_schema,
            'output_schema': self.output_schema
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(
            name=data['name'],
            version=data['version'],
            created_at=datetime.fromisoformat(data['created_at']),
            stage=ModelStage(data['stage']),
            metrics=data['metrics'],
            parameters=data['parameters'],
            tags=data['tags'],
            description=data['description'],
            artifact_path=data['artifact_path'],
            checksum=data['checksum'],
            framework=data.get('framework', 'unknown'),
            input_schema=data.get('input_schema'),
            output_schema=data.get('output_schema')
        )


@dataclass
class ModelVersion:
    """Container for a model version"""
    metadata: ModelMetadata
    model: Optional[Any] = None

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def stage(self) -> ModelStage:
        return self.metadata.stage

    @property
    def metrics(self) -> Dict[str, float]:
        return self.metadata.metrics


class ModelRegistry:
    """
    Model versioning and lifecycle management.

    Provides functionality to register, store, version, and load models.
    Tracks experiments and allows comparison between versions.

    Example:
    ```python
    registry = ModelRegistry('models/')

    # Register a new model
    version = registry.register(
        name='classifier',
        model=my_model,
        metrics={'accuracy': 0.95},
        tags=['experiment-1']
    )

    # Promote to production
    registry.transition_stage('classifier', version.version, ModelStage.PRODUCTION)

    # Load production model
    model = registry.load('classifier', stage=ModelStage.PRODUCTION)
    ```
    """

    def __init__(
        self,
        storage_path: Union[str, Path] = 'models',
        serialization: SerializationFormat = SerializationFormat.PICKLE
    ):
        """
        Initialize model registry.

        Args:
            storage_path: Directory for storing models and metadata
            serialization: Format for model serialization
        """
        self.storage_path = Path(storage_path)
        self.serialization = serialization
        self._models: Dict[str, Dict[str, ModelMetadata]] = {}  # name -> version -> metadata
        self._metadata_file = self.storage_path / 'registry.json'

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing registry
        self._load_registry()

    def _load_registry(self) -> None:
        """Load existing registry from disk"""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file, 'r') as f:
                    data = json.load(f)

                for name, versions in data.items():
                    self._models[name] = {}
                    for version, metadata_dict in versions.items():
                        self._models[name][version] = ModelMetadata.from_dict(metadata_dict)
            except Exception as e:
                print(f"Warning: Could not load registry: {e}")
                self._models = {}

    def _save_registry(self) -> None:
        """Save registry to disk"""
        data = {}
        for name, versions in self._models.items():
            data[name] = {}
            for version, metadata in versions.items():
                data[name][version] = metadata.to_dict()

        with open(self._metadata_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _generate_version(self, name: str) -> str:
        """Generate next version number"""
        if name not in self._models or not self._models[name]:
            return "1"

        versions = list(self._models[name].keys())
        try:
            max_version = max(int(v) for v in versions if v.isdigit())
            return str(max_version + 1)
        except ValueError:
            return str(len(versions) + 1)

    def _compute_checksum(self, model: Any) -> str:
        """Compute model checksum for integrity verification"""
        try:
            model_bytes = pickle.dumps(model)
            return hashlib.md5(model_bytes).hexdigest()
        except Exception:
            return "unknown"

    def _get_artifact_path(self, name: str, version: str) -> Path:
        """Get path for model artifact"""
        model_dir = self.storage_path / name
        model_dir.mkdir(parents=True, exist_ok=True)

        ext = {
            SerializationFormat.PICKLE: '.pkl',
            SerializationFormat.JOBLIB: '.joblib',
            SerializationFormat.JSON: '.json'
        }.get(self.serialization, '.pkl')

        return model_dir / f"{name}_v{version}{ext}"

    def register(
        self,
        name: str,
        model: Any,
        metrics: Optional[Dict[str, float]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        version: Optional[str] = None,
        stage: ModelStage = ModelStage.DEVELOPMENT,
        framework: str = "unknown",
        save_artifact: bool = True
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            name: Model name
            model: Model object to register
            metrics: Performance metrics
            parameters: Model parameters/hyperparameters
            tags: Tags for categorization
            description: Model description
            version: Explicit version (auto-generated if None)
            stage: Initial lifecycle stage
            framework: ML framework used (e.g., 'sklearn', 'tensorflow')
            save_artifact: Whether to save model to disk

        Returns:
            ModelVersion object
        """
        # Generate version if not provided
        if version is None:
            version = self._generate_version(name)

        # Compute checksum
        checksum = self._compute_checksum(model)

        # Get artifact path
        artifact_path = self._get_artifact_path(name, version)

        # Create metadata
        metadata = ModelMetadata(
            name=name,
            version=version,
            created_at=datetime.now(),
            stage=stage,
            metrics=metrics or {},
            parameters=parameters or {},
            tags=tags or [],
            description=description,
            artifact_path=str(artifact_path),
            checksum=checksum,
            framework=framework
        )

        # Save model artifact
        if save_artifact:
            self._save_model(model, artifact_path)

        # Register in memory
        if name not in self._models:
            self._models[name] = {}
        self._models[name][version] = metadata

        # Persist registry
        self._save_registry()

        return ModelVersion(metadata=metadata, model=model)

    def _save_model(self, model: Any, path: Path) -> None:
        """Save model to disk"""
        if self.serialization == SerializationFormat.PICKLE:
            with open(path, 'wb') as f:
                pickle.dump(model, f)
        elif self.serialization == SerializationFormat.JSON:
            with open(path, 'w') as f:
                if hasattr(model, 'to_dict'):
                    json.dump(model.to_dict(), f)
                else:
                    json.dump(model, f)
        elif self.serialization == SerializationFormat.JOBLIB:
            try:
                import joblib
                joblib.dump(model, path)
            except ImportError:
                # Fall back to pickle
                with open(path, 'wb') as f:
                    pickle.dump(model, f)

    def _load_model_artifact(self, path: Path) -> Any:
        """Load model from disk"""
        if not path.exists():
            raise FileNotFoundError(f"Model artifact not found: {path}")

        if self.serialization == SerializationFormat.PICKLE or str(path).endswith('.pkl'):
            with open(path, 'rb') as f:
                return pickle.load(f)
        elif self.serialization == SerializationFormat.JSON or str(path).endswith('.json'):
            with open(path, 'r') as f:
                return json.load(f)
        elif self.serialization == SerializationFormat.JOBLIB or str(path).endswith('.joblib'):
            try:
                import joblib
                return joblib.load(path)
            except ImportError:
                with open(path, 'rb') as f:
                    return pickle.load(f)

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> Any:
        """
        Load a model.

        Args:
            name: Model name
            version: Specific version (default: latest)
            stage: Load model at specific stage

        Returns:
            Loaded model object
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found in registry")

        if version == 'latest' or version is None:
            if stage is not None:
                # Find latest version at specified stage
                matching = [
                    (v, m) for v, m in self._models[name].items()
                    if m.stage == stage
                ]
                if not matching:
                    raise ValueError(f"No {stage.value} version found for '{name}'")
                version = max(matching, key=lambda x: x[1].created_at)[0]
            else:
                # Get latest version
                version = max(
                    self._models[name].keys(),
                    key=lambda v: self._models[name][v].created_at
                )

        if version not in self._models[name]:
            raise ValueError(f"Version '{version}' not found for model '{name}'")

        metadata = self._models[name][version]
        artifact_path = Path(metadata.artifact_path)

        return self._load_model_artifact(artifact_path)

    def load_version(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> ModelVersion:
        """
        Load model with metadata.

        Args:
            name: Model name
            version: Specific version
            stage: Load model at specific stage

        Returns:
            ModelVersion with model and metadata
        """
        model = self.load(name, version, stage)

        if version == 'latest' or version is None:
            if stage is not None:
                matching = [
                    (v, m) for v, m in self._models[name].items()
                    if m.stage == stage
                ]
                version = max(matching, key=lambda x: x[1].created_at)[0]
            else:
                version = max(
                    self._models[name].keys(),
                    key=lambda v: self._models[name][v].created_at
                )

        metadata = self._models[name][version]
        return ModelVersion(metadata=metadata, model=model)

    def transition_stage(
        self,
        name: str,
        version: str,
        stage: ModelStage
    ) -> ModelMetadata:
        """
        Transition model to a new lifecycle stage.

        Args:
            name: Model name
            version: Model version
            stage: Target stage

        Returns:
            Updated metadata
        """
        if name not in self._models or version not in self._models[name]:
            raise ValueError(f"Model '{name}' version '{version}' not found")

        # If promoting to production, archive current production
        if stage == ModelStage.PRODUCTION:
            for v, m in self._models[name].items():
                if m.stage == ModelStage.PRODUCTION:
                    m.stage = ModelStage.ARCHIVED

        self._models[name][version].stage = stage
        self._save_registry()

        return self._models[name][version]

    def get_latest_version(
        self,
        name: str,
        stage: Optional[ModelStage] = None
    ) -> Optional[str]:
        """
        Get latest version of a model.

        Args:
            name: Model name
            stage: Filter by stage

        Returns:
            Version string or None
        """
        if name not in self._models:
            return None

        if stage is not None:
            matching = [
                (v, m) for v, m in self._models[name].items()
                if m.stage == stage
            ]
            if not matching:
                return None
            return max(matching, key=lambda x: x[1].created_at)[0]
        else:
            return max(
                self._models[name].keys(),
                key=lambda v: self._models[name][v].created_at
            )

    def list_models(self) -> List[str]:
        """List all registered model names"""
        return list(self._models.keys())

    def list_versions(
        self,
        name: str,
        stage: Optional[ModelStage] = None
    ) -> List[ModelMetadata]:
        """
        List all versions of a model.

        Args:
            name: Model name
            stage: Filter by stage

        Returns:
            List of ModelMetadata
        """
        if name not in self._models:
            return []

        versions = list(self._models[name].values())

        if stage is not None:
            versions = [v for v in versions if v.stage == stage]

        return sorted(versions, key=lambda v: v.created_at, reverse=True)

    def get_production_model(self, name: str) -> Optional[ModelVersion]:
        """Get the current production model"""
        try:
            return self.load_version(name, stage=ModelStage.PRODUCTION)
        except ValueError:
            return None

    def compare_versions(
        self,
        name: str,
        versions: Optional[List[str]] = None,
        metric: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare model versions.

        Args:
            name: Model name
            versions: Specific versions to compare (default: all)
            metric: Specific metric to compare (default: all)

        Returns:
            Comparison results
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found")

        if versions is None:
            versions = list(self._models[name].keys())

        comparison = {
            'model': name,
            'versions': {},
            'best': {}
        }

        all_metrics = set()

        for v in versions:
            if v in self._models[name]:
                metadata = self._models[name][v]
                comparison['versions'][v] = {
                    'created_at': metadata.created_at.isoformat(),
                    'stage': metadata.stage.value,
                    'metrics': metadata.metrics,
                    'tags': metadata.tags
                }
                all_metrics.update(metadata.metrics.keys())

        # Find best version for each metric
        for m in all_metrics:
            if metric is not None and m != metric:
                continue

            best_version = None
            best_value = None

            for v in versions:
                if v in self._models[name]:
                    value = self._models[name][v].metrics.get(m)
                    if value is not None:
                        # Assume lower is better (typical for error metrics)
                        if best_value is None or value < best_value:
                            best_value = value
                            best_version = v

            if best_version:
                comparison['best'][m] = {
                    'version': best_version,
                    'value': best_value
                }

        return comparison

    def search_by_tags(self, tags: List[str], match_all: bool = False) -> List[ModelVersion]:
        """
        Search models by tags.

        Args:
            tags: Tags to search for
            match_all: If True, require all tags to match

        Returns:
            List of matching ModelVersion objects
        """
        results = []
        tags_set = set(tags)

        for name, versions in self._models.items():
            for version, metadata in versions.items():
                model_tags = set(metadata.tags)
                if match_all:
                    if tags_set.issubset(model_tags):
                        results.append(ModelVersion(metadata=metadata))
                else:
                    if tags_set.intersection(model_tags):
                        results.append(ModelVersion(metadata=metadata))

        return results

    def delete_version(self, name: str, version: str) -> bool:
        """
        Delete a specific model version.

        Args:
            name: Model name
            version: Version to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self._models or version not in self._models[name]:
            return False

        metadata = self._models[name][version]

        # Delete artifact
        artifact_path = Path(metadata.artifact_path)
        if artifact_path.exists():
            artifact_path.unlink()

        # Remove from registry
        del self._models[name][version]

        # Remove model entry if no versions left
        if not self._models[name]:
            del self._models[name]

        self._save_registry()
        return True

    def export_model(
        self,
        name: str,
        version: str,
        export_path: Union[str, Path],
        include_metadata: bool = True
    ) -> Path:
        """
        Export model to a specified path.

        Args:
            name: Model name
            version: Version to export
            export_path: Destination path
            include_metadata: Include metadata JSON file

        Returns:
            Path to exported model
        """
        if name not in self._models or version not in self._models[name]:
            raise ValueError(f"Model '{name}' version '{version}' not found")

        metadata = self._models[name][version]
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)

        # Copy model artifact
        import shutil
        src_path = Path(metadata.artifact_path)
        dest_path = export_path / src_path.name
        shutil.copy(src_path, dest_path)

        # Export metadata
        if include_metadata:
            metadata_path = export_path / f"{name}_v{version}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

        return dest_path

    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of all registered models"""
        summary = {
            'total_models': len(self._models),
            'total_versions': sum(len(v) for v in self._models.values()),
            'models': {}
        }

        for name, versions in self._models.items():
            prod_version = None
            for v, m in versions.items():
                if m.stage == ModelStage.PRODUCTION:
                    prod_version = v
                    break

            summary['models'][name] = {
                'version_count': len(versions),
                'latest_version': self.get_latest_version(name),
                'production_version': prod_version,
                'stages': {
                    stage.value: sum(1 for m in versions.values() if m.stage == stage)
                    for stage in ModelStage
                }
            }

        return summary


# ==================== Factory Functions ====================


def create_local_registry(path: str = 'models') -> ModelRegistry:
    """Create a local file-based model registry"""
    return ModelRegistry(storage_path=path, serialization=SerializationFormat.PICKLE)


def create_json_registry(path: str = 'models') -> ModelRegistry:
    """Create a registry using JSON serialization (for simple models)"""
    return ModelRegistry(storage_path=path, serialization=SerializationFormat.JSON)


# ==================== Exports ====================

__all__ = [
    'ModelRegistry',
    'ModelVersion',
    'ModelMetadata',
    'ModelStage',
    'SerializationFormat',
    'create_local_registry',
    'create_json_registry',
]
