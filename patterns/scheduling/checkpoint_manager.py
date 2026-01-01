"""
Checkpoint Manager - Universal Business Solution Framework

Checkpoint and resume functionality for long-running processes.
Enables recovery from failures and incremental processing.

Example:
```python
from patterns.scheduling import CheckpointManager

# Create checkpoint manager
checkpoint = CheckpointManager("data_processing", storage_path="checkpoints/")

# Resume or start fresh
state = checkpoint.load() or {"processed": 0, "items": []}

# Process with checkpointing
for i, item in enumerate(items):
    if i < state["processed"]:
        continue  # Skip already processed

    process(item)
    state["processed"] = i + 1

    # Save checkpoint periodically
    if i % 100 == 0:
        checkpoint.save(state)

# Final save
checkpoint.save(state, complete=True)
```
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, TypeVar, Generic
import json
import hashlib
import threading
from pathlib import Path
import shutil


class CheckpointStatus(Enum):
    """Checkpoint status."""
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass
class CheckpointMetadata:
    """Metadata about a checkpoint."""
    checkpoint_id: str
    job_name: str
    created_at: datetime
    updated_at: datetime
    status: CheckpointStatus
    version: int = 1
    items_processed: int = 0
    total_items: Optional[int] = None
    error: Optional[str] = None
    custom: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        if self.total_items and self.total_items > 0:
            return (self.items_processed / self.total_items) * 100
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "job_name": self.job_name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "version": self.version,
            "items_processed": self.items_processed,
            "total_items": self.total_items,
            "error": self.error,
            "custom": self.custom
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        return cls(
            checkpoint_id=data["checkpoint_id"],
            job_name=data["job_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            status=CheckpointStatus(data["status"]),
            version=data.get("version", 1),
            items_processed=data.get("items_processed", 0),
            total_items=data.get("total_items"),
            error=data.get("error"),
            custom=data.get("custom", {})
        )


class CheckpointManager:
    """
    Manage checkpoints for resumable processing.

    Features:
    - Save and load state
    - Automatic versioning
    - Progress tracking
    - Checkpoint history
    - Expiration handling

    Example:
    ```python
    checkpoint = CheckpointManager("etl_job", storage_path="checkpoints/")

    # Check for existing checkpoint
    if checkpoint.exists():
        state = checkpoint.load()
        print(f"Resuming from item {state['last_id']}")
    else:
        state = {"last_id": 0, "errors": []}

    # Process with periodic saves
    for item in get_items(after=state["last_id"]):
        try:
            process(item)
            state["last_id"] = item.id
        except Exception as e:
            state["errors"].append(str(e))

        # Save every 100 items
        if item.id % 100 == 0:
            checkpoint.save(state)

    # Mark complete
    checkpoint.complete(state)
    ```
    """

    def __init__(
        self,
        job_name: str,
        storage_path: str = "checkpoints",
        max_history: int = 5,
        expiration_hours: int = 24,
        auto_save_interval: Optional[int] = None
    ):
        """
        Initialize checkpoint manager.

        Args:
            job_name: Name of the job/process
            storage_path: Directory for checkpoint files
            max_history: Max checkpoint versions to keep
            expiration_hours: Hours until checkpoint expires
            auto_save_interval: Auto-save every N operations
        """
        self.job_name = job_name
        self.storage_path = Path(storage_path)
        self.max_history = max_history
        self.expiration_hours = expiration_hours
        self.auto_save_interval = auto_save_interval

        self._lock = threading.Lock()
        self._operation_count = 0
        self._current_state: Optional[Dict[str, Any]] = None

        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_file(self) -> Path:
        """Get current checkpoint file path."""
        return self.storage_path / f"{self.job_name}_checkpoint.json"

    @property
    def metadata_file(self) -> Path:
        """Get metadata file path."""
        return self.storage_path / f"{self.job_name}_metadata.json"

    def exists(self) -> bool:
        """Check if a valid checkpoint exists."""
        if not self.checkpoint_file.exists():
            return False

        # Check expiration
        metadata = self._load_metadata()
        if metadata:
            if metadata.status == CheckpointStatus.COMPLETED:
                return False  # Completed checkpoints should be restarted
            if metadata.status == CheckpointStatus.EXPIRED:
                return False

            # Check if expired
            age = datetime.now() - metadata.updated_at
            if age > timedelta(hours=self.expiration_hours):
                self._update_metadata_status(CheckpointStatus.EXPIRED)
                return False

        return True

    def load(self) -> Optional[Dict[str, Any]]:
        """
        Load the most recent checkpoint state.

        Returns:
            Checkpoint state dict or None if no valid checkpoint
        """
        if not self.exists():
            return None

        with self._lock:
            try:
                content = self.checkpoint_file.read_text(encoding='utf-8')
                data = json.loads(content)
                self._current_state = data.get("state", {})
                return self._current_state
            except Exception:
                return None

    def save(
        self,
        state: Dict[str, Any],
        items_processed: Optional[int] = None,
        total_items: Optional[int] = None,
        complete: bool = False
    ) -> bool:
        """
        Save checkpoint state.

        Args:
            state: State dict to save
            items_processed: Number of items processed
            total_items: Total items to process
            complete: Mark as completed

        Returns:
            True if saved successfully
        """
        with self._lock:
            try:
                # Load or create metadata
                metadata = self._load_metadata()
                now = datetime.now()

                if metadata:
                    metadata.updated_at = now
                    metadata.version += 1
                    if items_processed is not None:
                        metadata.items_processed = items_processed
                    if total_items is not None:
                        metadata.total_items = total_items
                    if complete:
                        metadata.status = CheckpointStatus.COMPLETED
                else:
                    metadata = CheckpointMetadata(
                        checkpoint_id=self._generate_id(),
                        job_name=self.job_name,
                        created_at=now,
                        updated_at=now,
                        status=CheckpointStatus.COMPLETED if complete else CheckpointStatus.ACTIVE,
                        items_processed=items_processed or 0,
                        total_items=total_items
                    )

                # Create backup of previous checkpoint
                if self.checkpoint_file.exists():
                    self._create_backup()

                # Save state
                checkpoint_data = {
                    "state": state,
                    "metadata": metadata.to_dict()
                }

                self.checkpoint_file.write_text(
                    json.dumps(checkpoint_data, indent=2, default=str),
                    encoding='utf-8'
                )

                # Save metadata separately
                self.metadata_file.write_text(
                    json.dumps(metadata.to_dict(), indent=2),
                    encoding='utf-8'
                )

                self._current_state = state
                return True

            except Exception:
                return False

    def complete(self, state: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark checkpoint as completed.

        Args:
            state: Optional final state to save

        Returns:
            True if marked successfully
        """
        final_state = state or self._current_state or {}
        return self.save(final_state, complete=True)

    def fail(self, error: str, state: Optional[Dict[str, Any]] = None) -> bool:
        """
        Mark checkpoint as failed.

        Args:
            error: Error message
            state: Optional state at failure

        Returns:
            True if marked successfully
        """
        with self._lock:
            metadata = self._load_metadata()
            if metadata:
                metadata.status = CheckpointStatus.FAILED
                metadata.error = error
                metadata.updated_at = datetime.now()

                self.metadata_file.write_text(
                    json.dumps(metadata.to_dict(), indent=2),
                    encoding='utf-8'
                )

            if state:
                self.save(state)

            return True

    def clear(self) -> bool:
        """
        Remove current checkpoint.

        Returns:
            True if cleared successfully
        """
        with self._lock:
            try:
                if self.checkpoint_file.exists():
                    self.checkpoint_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                self._current_state = None
                return True
            except Exception:
                return False

    def get_metadata(self) -> Optional[CheckpointMetadata]:
        """Get checkpoint metadata."""
        return self._load_metadata()

    def get_progress(self) -> Dict[str, Any]:
        """
        Get checkpoint progress info.

        Returns:
            Dict with progress information
        """
        metadata = self._load_metadata()
        if not metadata:
            return {
                "exists": False,
                "progress": 0,
                "items_processed": 0,
                "total_items": None
            }

        return {
            "exists": True,
            "status": metadata.status.value,
            "progress": metadata.progress,
            "items_processed": metadata.items_processed,
            "total_items": metadata.total_items,
            "created_at": metadata.created_at.isoformat(),
            "updated_at": metadata.updated_at.isoformat(),
            "version": metadata.version
        }

    def increment(self, count: int = 1) -> Optional[Dict[str, Any]]:
        """
        Increment operation count and auto-save if needed.

        Args:
            count: Number of operations to add

        Returns:
            Current state if auto-saved, None otherwise
        """
        self._operation_count += count

        if self.auto_save_interval and self._operation_count >= self.auto_save_interval:
            if self._current_state:
                self.save(self._current_state, items_processed=self._operation_count)
                self._operation_count = 0
                return self._current_state

        return None

    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get checkpoint history (previous versions).

        Returns:
            List of previous checkpoint metadata
        """
        history = []
        backup_pattern = f"{self.job_name}_checkpoint_*.json"

        for backup_file in sorted(self.storage_path.glob(backup_pattern), reverse=True):
            try:
                content = backup_file.read_text(encoding='utf-8')
                data = json.loads(content)
                if "metadata" in data:
                    history.append(data["metadata"])
            except Exception:
                continue

        return history[:self.max_history]

    def restore(self, version: int) -> Optional[Dict[str, Any]]:
        """
        Restore checkpoint from a specific version.

        Args:
            version: Version number to restore

        Returns:
            Restored state or None
        """
        for backup_file in self.storage_path.glob(f"{self.job_name}_checkpoint_*.json"):
            try:
                content = backup_file.read_text(encoding='utf-8')
                data = json.loads(content)
                if data.get("metadata", {}).get("version") == version:
                    # Copy backup to current
                    self.checkpoint_file.write_text(content, encoding='utf-8')
                    return data.get("state")
            except Exception:
                continue

        return None

    def _load_metadata(self) -> Optional[CheckpointMetadata]:
        """Load metadata from file."""
        if not self.metadata_file.exists():
            return None

        try:
            content = self.metadata_file.read_text(encoding='utf-8')
            return CheckpointMetadata.from_dict(json.loads(content))
        except Exception:
            return None

    def _update_metadata_status(self, status: CheckpointStatus):
        """Update metadata status."""
        metadata = self._load_metadata()
        if metadata:
            metadata.status = status
            metadata.updated_at = datetime.now()
            self.metadata_file.write_text(
                json.dumps(metadata.to_dict(), indent=2),
                encoding='utf-8'
            )

    def _create_backup(self):
        """Create backup of current checkpoint."""
        if not self.checkpoint_file.exists():
            return

        # Get current version
        metadata = self._load_metadata()
        version = metadata.version if metadata else 0

        # Copy to backup
        backup_name = f"{self.job_name}_checkpoint_{version:04d}.json"
        backup_path = self.storage_path / backup_name
        shutil.copy2(self.checkpoint_file, backup_path)

        # Clean old backups
        self._cleanup_old_backups()

    def _cleanup_old_backups(self):
        """Remove old backup files."""
        backups = sorted(
            self.storage_path.glob(f"{self.job_name}_checkpoint_*.json"),
            reverse=True
        )

        for old_backup in backups[self.max_history:]:
            try:
                old_backup.unlink()
            except Exception:
                pass

    def _generate_id(self) -> str:
        """Generate unique checkpoint ID."""
        content = f"{self.job_name}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


class IncrementalProcessor:
    """
    Process items incrementally with automatic checkpointing.

    Example:
    ```python
    processor = IncrementalProcessor(
        "sync_users",
        storage_path="checkpoints/"
    )

    def process_user(user, state):
        # Process user
        sync_user(user)
        state["synced_count"] += 1
        return state

    # Resume or start fresh
    processor.process(
        items=get_users(),
        process_func=process_user,
        initial_state={"synced_count": 0}
    )
    ```
    """

    def __init__(
        self,
        job_name: str,
        storage_path: str = "checkpoints",
        batch_size: int = 100,
        checkpoint_interval: int = 100
    ):
        """
        Initialize incremental processor.

        Args:
            job_name: Name of the job
            storage_path: Checkpoint storage path
            batch_size: Items per batch
            checkpoint_interval: Checkpoint every N items
        """
        self.checkpoint = CheckpointManager(
            job_name,
            storage_path=storage_path,
            auto_save_interval=checkpoint_interval
        )
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval

    def process(
        self,
        items: List[Any],
        process_func: Callable[[Any, Dict], Dict],
        initial_state: Optional[Dict[str, Any]] = None,
        item_id_func: Optional[Callable[[Any], Any]] = None,
        on_error: str = "skip"  # "skip", "stop", "retry"
    ) -> Dict[str, Any]:
        """
        Process items with checkpointing.

        Args:
            items: Items to process
            process_func: Function(item, state) -> state
            initial_state: Initial state if no checkpoint
            item_id_func: Function to get item ID for tracking
            on_error: Error handling strategy

        Returns:
            Final state
        """
        # Load or initialize state
        state = self.checkpoint.load()
        if state is None:
            state = initial_state or {}
            state["_processed_ids"] = set()
            state["_error_count"] = 0
        else:
            state["_processed_ids"] = set(state.get("_processed_ids", []))

        total = len(items)

        try:
            for i, item in enumerate(items):
                # Get item ID
                item_id = item_id_func(item) if item_id_func else i

                # Skip already processed
                if item_id in state["_processed_ids"]:
                    continue

                try:
                    # Process item
                    state = process_func(item, state)
                    state["_processed_ids"].add(item_id)

                except Exception as e:
                    state["_error_count"] = state.get("_error_count", 0) + 1

                    if on_error == "stop":
                        self.checkpoint.fail(str(e), state)
                        raise
                    elif on_error == "skip":
                        state["_processed_ids"].add(item_id)  # Mark as processed
                    # "retry" - don't mark as processed

                # Checkpoint periodically
                if (i + 1) % self.checkpoint_interval == 0:
                    # Convert set to list for JSON serialization
                    save_state = {**state, "_processed_ids": list(state["_processed_ids"])}
                    self.checkpoint.save(save_state, items_processed=i + 1, total_items=total)

            # Final save
            save_state = {**state, "_processed_ids": list(state["_processed_ids"])}
            self.checkpoint.complete(save_state)

        except Exception:
            save_state = {**state, "_processed_ids": list(state["_processed_ids"])}
            self.checkpoint.save(save_state)
            raise

        return state


# Factory functions

def create_checkpoint_manager(
    job_name: str,
    storage_path: str = "checkpoints"
) -> CheckpointManager:
    """Create a basic checkpoint manager."""
    return CheckpointManager(job_name, storage_path)


def create_auto_checkpoint(
    job_name: str,
    interval: int = 100,
    storage_path: str = "checkpoints"
) -> CheckpointManager:
    """
    Create checkpoint manager with auto-save.

    Args:
        job_name: Job name
        interval: Save every N operations
        storage_path: Storage path

    Returns:
        Configured CheckpointManager
    """
    return CheckpointManager(
        job_name,
        storage_path=storage_path,
        auto_save_interval=interval
    )


def create_incremental_processor(
    job_name: str,
    batch_size: int = 100
) -> IncrementalProcessor:
    """
    Create incremental processor.

    Args:
        job_name: Job name
        batch_size: Items per batch

    Returns:
        Configured IncrementalProcessor
    """
    return IncrementalProcessor(
        job_name,
        batch_size=batch_size,
        checkpoint_interval=batch_size
    )
