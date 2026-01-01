"""
Pipeline Orchestrator - Universal Business Solution Framework

DAG-based task pipeline execution with dependency management,
parallel execution, and error handling.

Example:
```python
from patterns.scheduling import Pipeline, Task, TaskStatus

# Create pipeline
pipeline = Pipeline("data_pipeline")

# Add tasks with dependencies
pipeline.add_task(Task("extract", extract_func))
pipeline.add_task(Task("transform", transform_func, depends_on=["extract"]))
pipeline.add_task(Task("load", load_func, depends_on=["transform"]))
pipeline.add_task(Task("validate", validate_func, depends_on=["load"]))

# Run pipeline
result = pipeline.run()
print(f"Status: {result.status}")
print(f"Duration: {result.duration_ms}ms")
```
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
import json
from pathlib import Path


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class PipelineStatus(Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"  # Some tasks failed


@dataclass
class TaskResult:
    """Result of task execution."""
    task_name: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: float = 0
    retries: int = 0


@dataclass
class Task:
    """
    Pipeline task definition.

    Example:
    ```python
    task = Task(
        name="process_data",
        func=process_function,
        depends_on=["fetch_data"],
        args=[arg1],
        kwargs={"key": "value"},
        max_retries=3
    )
    ```
    """
    name: str
    func: Callable
    depends_on: List[str] = field(default_factory=list)
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    max_retries: int = 0
    retry_delay_seconds: float = 1.0
    timeout_seconds: Optional[float] = None
    on_failure: str = "fail"  # "fail", "skip", "continue"
    tags: List[str] = field(default_factory=list)
    description: str = ""

    # Runtime state
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[TaskResult] = None


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    pipeline_name: str
    status: PipelineStatus
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: float = 0
    task_results: Dict[str, TaskResult] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def completed_count(self) -> int:
        return sum(1 for r in self.task_results.values() if r.status == TaskStatus.COMPLETED)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.task_results.values() if r.status == TaskStatus.FAILED)

    @property
    def success_rate(self) -> float:
        if not self.task_results:
            return 0.0
        return self.completed_count / len(self.task_results)


class Pipeline:
    """
    DAG-based task pipeline.

    Features:
    - Dependency-based execution order
    - Parallel execution of independent tasks
    - Retry logic per task
    - Failure handling strategies
    - Execution context sharing

    Example:
    ```python
    pipeline = Pipeline("etl_pipeline", max_parallel=4)

    # Add tasks
    pipeline.add_task(Task("extract_a", extract_a))
    pipeline.add_task(Task("extract_b", extract_b))
    pipeline.add_task(Task("transform", transform, depends_on=["extract_a", "extract_b"]))
    pipeline.add_task(Task("load", load, depends_on=["transform"]))

    # Run
    result = pipeline.run()

    # Check results
    for task_name, task_result in result.task_results.items():
        print(f"{task_name}: {task_result.status.value}")
    ```
    """

    def __init__(
        self,
        name: str,
        max_parallel: int = 4,
        stop_on_failure: bool = True,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize pipeline.

        Args:
            name: Pipeline name
            max_parallel: Max parallel tasks
            stop_on_failure: Stop pipeline on task failure
            context: Shared context dict passed to all tasks
        """
        self.name = name
        self.max_parallel = max_parallel
        self.stop_on_failure = stop_on_failure
        self.context = context or {}

        self._tasks: Dict[str, Task] = {}
        self._lock = threading.Lock()
        self._cancelled = False

    def add_task(self, task: Task) -> 'Pipeline':
        """
        Add a task to the pipeline.

        Args:
            task: Task to add

        Returns:
            Self for chaining
        """
        with self._lock:
            self._tasks[task.name] = task
        return self

    def add_tasks(self, tasks: List[Task]) -> 'Pipeline':
        """Add multiple tasks."""
        for task in tasks:
            self.add_task(task)
        return self

    def remove_task(self, name: str) -> bool:
        """Remove a task from the pipeline."""
        with self._lock:
            if name in self._tasks:
                del self._tasks[name]
                return True
            return False

    def get_task(self, name: str) -> Optional[Task]:
        """Get a task by name."""
        return self._tasks.get(name)

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate pipeline configuration.

        Returns:
            Tuple of (is_valid, list of errors)
        """
        errors = []

        # Check for missing dependencies
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep not in self._tasks:
                    errors.append(f"Task '{task.name}' depends on missing task '{dep}'")

        # Check for circular dependencies
        if not errors:
            try:
                self._topological_sort()
            except ValueError as e:
                errors.append(str(e))

        return len(errors) == 0, errors

    def _topological_sort(self) -> List[str]:
        """
        Sort tasks in topological order.

        Returns:
            List of task names in execution order

        Raises:
            ValueError if circular dependency detected
        """
        in_degree = {name: 0 for name in self._tasks}
        for task in self._tasks.values():
            for dep in task.depends_on:
                if dep in in_degree:
                    in_degree[task.name] += 1

        # Start with tasks that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            name = queue.pop(0)
            result.append(name)

            # Reduce in-degree for dependent tasks
            for task in self._tasks.values():
                if name in task.depends_on:
                    in_degree[task.name] -= 1
                    if in_degree[task.name] == 0:
                        queue.append(task.name)

        if len(result) != len(self._tasks):
            raise ValueError("Circular dependency detected in pipeline")

        return result

    def get_execution_order(self) -> List[List[str]]:
        """
        Get execution order showing parallel groups.

        Returns:
            List of lists, where each inner list can run in parallel
        """
        completed: Set[str] = set()
        order = []

        while len(completed) < len(self._tasks):
            # Find tasks whose dependencies are all completed
            ready = []
            for name, task in self._tasks.items():
                if name in completed:
                    continue
                if all(dep in completed for dep in task.depends_on):
                    ready.append(name)

            if not ready:
                break

            order.append(ready)
            completed.update(ready)

        return order

    def run(
        self,
        context: Optional[Dict[str, Any]] = None,
        resume_from: Optional[str] = None
    ) -> PipelineResult:
        """
        Execute the pipeline.

        Args:
            context: Additional context to merge
            resume_from: Resume from specific task (skip completed)

        Returns:
            PipelineResult with all task results
        """
        self._cancelled = False

        # Merge context
        if context:
            self.context.update(context)

        # Validate
        is_valid, errors = self.validate()
        if not is_valid:
            return PipelineResult(
                pipeline_name=self.name,
                status=PipelineStatus.FAILED,
                started_at=datetime.now(),
                ended_at=datetime.now(),
                error=f"Validation failed: {'; '.join(errors)}"
            )

        result = PipelineResult(
            pipeline_name=self.name,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now()
        )

        # Reset task states
        for task in self._tasks.values():
            task.status = TaskStatus.PENDING
            task.result = None

        # Get execution order
        execution_order = self.get_execution_order()

        # Skip tasks if resuming
        skip_remaining = resume_from is not None
        completed_tasks: Set[str] = set()
        failed = False

        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            for group in execution_order:
                if self._cancelled:
                    break

                # Handle resume
                if skip_remaining:
                    if resume_from in group:
                        skip_remaining = False
                        # Only run from resume point in this group
                        group = group[group.index(resume_from):]
                    else:
                        # Skip entire group
                        for name in group:
                            task_result = TaskResult(
                                task_name=name,
                                status=TaskStatus.SKIPPED,
                                started_at=datetime.now(),
                                ended_at=datetime.now()
                            )
                            result.task_results[name] = task_result
                            completed_tasks.add(name)
                        continue

                # Submit all tasks in group
                futures: Dict[Future, str] = {}
                for task_name in group:
                    if task_name in completed_tasks:
                        continue

                    task = self._tasks[task_name]
                    task.status = TaskStatus.QUEUED

                    future = executor.submit(self._execute_task, task)
                    futures[future] = task_name

                # Wait for all in group
                for future in as_completed(futures):
                    task_name = futures[future]
                    task = self._tasks[task_name]

                    try:
                        task_result = future.result()
                    except Exception as e:
                        task_result = TaskResult(
                            task_name=task_name,
                            status=TaskStatus.FAILED,
                            error=str(e),
                            ended_at=datetime.now()
                        )

                    result.task_results[task_name] = task_result
                    task.result = task_result

                    if task_result.status == TaskStatus.COMPLETED:
                        completed_tasks.add(task_name)
                    elif task_result.status == TaskStatus.FAILED:
                        if self.stop_on_failure and task.on_failure == "fail":
                            failed = True

                # Check if should stop
                if failed and self.stop_on_failure:
                    # Mark remaining as skipped
                    for remaining_group in execution_order[execution_order.index(group) + 1:]:
                        for name in remaining_group:
                            if name not in result.task_results:
                                result.task_results[name] = TaskResult(
                                    task_name=name,
                                    status=TaskStatus.SKIPPED
                                )
                    break

        # Finalize result
        result.ended_at = datetime.now()
        result.duration_ms = (result.ended_at - result.started_at).total_seconds() * 1000

        # Determine final status
        if self._cancelled:
            result.status = PipelineStatus.CANCELLED
        elif result.failed_count > 0:
            if result.completed_count > 0:
                result.status = PipelineStatus.PARTIAL
            else:
                result.status = PipelineStatus.FAILED
        else:
            result.status = PipelineStatus.COMPLETED

        return result

    def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task with retry logic."""
        task.status = TaskStatus.RUNNING

        result = TaskResult(
            task_name=task.name,
            status=TaskStatus.RUNNING,
            started_at=datetime.now()
        )

        retries = 0
        while retries <= task.max_retries:
            try:
                # Build args (inject context if function accepts it)
                kwargs = {**task.kwargs}

                # Check if function wants context
                import inspect
                sig = inspect.signature(task.func)
                if 'context' in sig.parameters:
                    kwargs['context'] = self.context
                if 'pipeline' in sig.parameters:
                    kwargs['pipeline'] = self

                # Execute
                output = task.func(*task.args, **kwargs)

                result.status = TaskStatus.COMPLETED
                result.result = output
                task.status = TaskStatus.COMPLETED
                break

            except Exception as e:
                retries += 1
                result.retries = retries

                if retries <= task.max_retries:
                    time.sleep(task.retry_delay_seconds * retries)
                else:
                    result.status = TaskStatus.FAILED
                    result.error = str(e)
                    task.status = TaskStatus.FAILED

        result.ended_at = datetime.now()
        result.duration_ms = (result.ended_at - result.started_at).total_seconds() * 1000

        return result

    def cancel(self):
        """Cancel pipeline execution."""
        self._cancelled = True

    def visualize(self) -> str:
        """
        Generate ASCII visualization of pipeline.

        Returns:
            ASCII diagram string
        """
        execution_order = self.get_execution_order()
        lines = [f"Pipeline: {self.name}", "=" * (len(self.name) + 10)]

        for i, group in enumerate(execution_order):
            if i > 0:
                lines.append("    |")
                lines.append("    v")

            if len(group) == 1:
                lines.append(f"  [{group[0]}]")
            else:
                parallel_line = " + ".join(f"[{name}]" for name in group)
                lines.append(f"  {parallel_line}")
                lines.append(f"  (parallel)")

        return "\n".join(lines)


class PipelineBuilder:
    """
    Fluent builder for creating pipelines.

    Example:
    ```python
    pipeline = (PipelineBuilder("etl")
        .add("extract", extract_func)
        .add("transform", transform_func, depends_on=["extract"])
        .add("load", load_func, depends_on=["transform"])
        .with_max_parallel(4)
        .build())
    ```
    """

    def __init__(self, name: str):
        self.name = name
        self._tasks: List[Task] = []
        self._max_parallel = 4
        self._stop_on_failure = True
        self._context: Dict[str, Any] = {}

    def add(
        self,
        name: str,
        func: Callable,
        depends_on: Optional[List[str]] = None,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        max_retries: int = 0
    ) -> 'PipelineBuilder':
        """Add a task."""
        self._tasks.append(Task(
            name=name,
            func=func,
            depends_on=depends_on or [],
            args=args or [],
            kwargs=kwargs or {},
            max_retries=max_retries
        ))
        return self

    def with_max_parallel(self, max_parallel: int) -> 'PipelineBuilder':
        """Set max parallel tasks."""
        self._max_parallel = max_parallel
        return self

    def with_stop_on_failure(self, stop: bool) -> 'PipelineBuilder':
        """Set stop on failure behavior."""
        self._stop_on_failure = stop
        return self

    def with_context(self, context: Dict[str, Any]) -> 'PipelineBuilder':
        """Set pipeline context."""
        self._context = context
        return self

    def build(self) -> Pipeline:
        """Build the pipeline."""
        pipeline = Pipeline(
            name=self.name,
            max_parallel=self._max_parallel,
            stop_on_failure=self._stop_on_failure,
            context=self._context
        )

        for task in self._tasks:
            pipeline.add_task(task)

        return pipeline


class ETLPipeline(Pipeline):
    """
    Specialized pipeline for ETL workflows.

    Provides standard extract, transform, load structure.

    Example:
    ```python
    etl = ETLPipeline("user_sync")
    etl.set_extract(fetch_users)
    etl.add_transform("clean", clean_data)
    etl.add_transform("enrich", enrich_data)
    etl.set_load(save_to_db)

    result = etl.run()
    ```
    """

    def __init__(self, name: str, max_parallel: int = 4):
        super().__init__(name, max_parallel)
        self._extract_task: Optional[Task] = None
        self._transform_tasks: List[Task] = []
        self._load_task: Optional[Task] = None

    def set_extract(
        self,
        func: Callable,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> 'ETLPipeline':
        """Set extract task."""
        self._extract_task = Task(
            name="extract",
            func=func,
            args=args or [],
            kwargs=kwargs or {}
        )
        return self

    def add_transform(
        self,
        name: str,
        func: Callable,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        depends_on_transform: Optional[str] = None
    ) -> 'ETLPipeline':
        """Add transform task."""
        depends = ["extract"]
        if depends_on_transform:
            depends = [f"transform_{depends_on_transform}"]
        elif self._transform_tasks:
            depends = [self._transform_tasks[-1].name]

        task = Task(
            name=f"transform_{name}",
            func=func,
            depends_on=depends,
            args=args or [],
            kwargs=kwargs or {}
        )
        self._transform_tasks.append(task)
        return self

    def set_load(
        self,
        func: Callable,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None
    ) -> 'ETLPipeline':
        """Set load task."""
        depends = []
        if self._transform_tasks:
            depends = [self._transform_tasks[-1].name]
        elif self._extract_task:
            depends = ["extract"]

        self._load_task = Task(
            name="load",
            func=func,
            depends_on=depends,
            args=args or [],
            kwargs=kwargs or {}
        )
        return self

    def run(self, **kwargs) -> PipelineResult:
        """Run the ETL pipeline."""
        # Build tasks
        self._tasks.clear()

        if self._extract_task:
            self.add_task(self._extract_task)

        for task in self._transform_tasks:
            self.add_task(task)

        if self._load_task:
            self.add_task(self._load_task)

        return super().run(**kwargs)


# Factory functions

def create_simple_pipeline(name: str, tasks: List[Tuple[str, Callable]]) -> Pipeline:
    """
    Create a simple linear pipeline.

    Args:
        name: Pipeline name
        tasks: List of (name, func) tuples

    Returns:
        Linear Pipeline
    """
    pipeline = Pipeline(name)
    prev_name = None

    for task_name, func in tasks:
        depends = [prev_name] if prev_name else []
        pipeline.add_task(Task(name=task_name, func=func, depends_on=depends))
        prev_name = task_name

    return pipeline


def create_parallel_pipeline(
    name: str,
    parallel_tasks: List[Tuple[str, Callable]],
    final_task: Optional[Tuple[str, Callable]] = None
) -> Pipeline:
    """
    Create a pipeline with parallel tasks and optional final aggregator.

    Args:
        name: Pipeline name
        parallel_tasks: Tasks to run in parallel
        final_task: Optional final task after parallel tasks

    Returns:
        Parallel Pipeline
    """
    pipeline = Pipeline(name)
    parallel_names = []

    for task_name, func in parallel_tasks:
        pipeline.add_task(Task(name=task_name, func=func))
        parallel_names.append(task_name)

    if final_task:
        name, func = final_task
        pipeline.add_task(Task(name=name, func=func, depends_on=parallel_names))

    return pipeline
