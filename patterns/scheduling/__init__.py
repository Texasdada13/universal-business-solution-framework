"""
Scheduling Patterns - Universal Business Solution Framework

Job scheduling, pipeline orchestration, and checkpoint/resume functionality.
Supports cron-like scheduling, DAG-based pipelines, and incremental processing.

Usage:
```python
from patterns.scheduling import (
    # Job Scheduler
    JobScheduler, Job, Schedule, JobStatus,
    SimpleScheduler,
    create_simple_scheduler, create_monitored_scheduler,

    # Pipeline Orchestrator
    Pipeline, Task, TaskStatus, PipelineResult,
    PipelineBuilder, ETLPipeline,
    create_simple_pipeline, create_parallel_pipeline,

    # Checkpoint Manager
    CheckpointManager, CheckpointStatus,
    IncrementalProcessor,
    create_checkpoint_manager, create_auto_checkpoint,
)
```
"""

from .job_scheduler import (
    JobScheduler,
    Job,
    Schedule,
    ScheduleType,
    JobStatus,
    JobExecution,
    SimpleScheduler,
    # Factory functions
    create_simple_scheduler,
    create_monitored_scheduler,
    create_sequential_scheduler,
)

from .pipeline_orchestrator import (
    Pipeline,
    Task,
    TaskStatus,
    TaskResult,
    PipelineStatus,
    PipelineResult,
    PipelineBuilder,
    ETLPipeline,
    # Factory functions
    create_simple_pipeline,
    create_parallel_pipeline,
)

from .checkpoint_manager import (
    CheckpointManager,
    CheckpointStatus,
    CheckpointMetadata,
    IncrementalProcessor,
    # Factory functions
    create_checkpoint_manager,
    create_auto_checkpoint,
    create_incremental_processor,
)

__all__ = [
    # Job Scheduler
    "JobScheduler",
    "Job",
    "Schedule",
    "ScheduleType",
    "JobStatus",
    "JobExecution",
    "SimpleScheduler",
    "create_simple_scheduler",
    "create_monitored_scheduler",
    "create_sequential_scheduler",

    # Pipeline Orchestrator
    "Pipeline",
    "Task",
    "TaskStatus",
    "TaskResult",
    "PipelineStatus",
    "PipelineResult",
    "PipelineBuilder",
    "ETLPipeline",
    "create_simple_pipeline",
    "create_parallel_pipeline",

    # Checkpoint Manager
    "CheckpointManager",
    "CheckpointStatus",
    "CheckpointMetadata",
    "IncrementalProcessor",
    "create_checkpoint_manager",
    "create_auto_checkpoint",
    "create_incremental_processor",
]
