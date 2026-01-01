"""
Job Scheduler - Universal Business Solution Framework

Cron-like job scheduling with dependency tracking, retry logic, and monitoring.
Supports interval, cron expression, and one-time scheduling.

Example:
```python
from patterns.scheduling import JobScheduler, Job, Schedule

# Create scheduler
scheduler = JobScheduler()

# Add jobs with different schedules
scheduler.add_job(Job(
    name="daily_report",
    func=generate_report,
    schedule=Schedule.daily(hour=8, minute=0)
))

scheduler.add_job(Job(
    name="hourly_sync",
    func=sync_data,
    schedule=Schedule.hourly()
))

scheduler.add_job(Job(
    name="every_5_min",
    func=check_alerts,
    schedule=Schedule.interval(minutes=5)
))

# Start scheduler
scheduler.start()
```
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
import threading
import time
import logging
from pathlib import Path
import json


class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class ScheduleType(Enum):
    """Schedule type."""
    INTERVAL = "interval"
    CRON = "cron"
    DAILY = "daily"
    HOURLY = "hourly"
    ONCE = "once"


@dataclass
class Schedule:
    """
    Job schedule definition.

    Factory methods for common schedules:
    - Schedule.interval(minutes=5)
    - Schedule.daily(hour=8, minute=0)
    - Schedule.hourly(minute=0)
    - Schedule.cron("0 8 * * 1-5")  # Weekdays at 8am
    - Schedule.once(datetime(2024, 1, 1))
    """
    schedule_type: ScheduleType
    interval_seconds: int = 0
    cron_expression: str = ""
    hour: int = 0
    minute: int = 0
    day_of_week: Optional[List[int]] = None  # 0=Monday, 6=Sunday
    run_at: Optional[datetime] = None

    @classmethod
    def interval(cls, seconds: int = 0, minutes: int = 0, hours: int = 0) -> 'Schedule':
        """Create interval schedule."""
        total_seconds = seconds + (minutes * 60) + (hours * 3600)
        return cls(schedule_type=ScheduleType.INTERVAL, interval_seconds=total_seconds)

    @classmethod
    def daily(cls, hour: int = 0, minute: int = 0, days: Optional[List[int]] = None) -> 'Schedule':
        """Create daily schedule at specific time."""
        return cls(
            schedule_type=ScheduleType.DAILY,
            hour=hour,
            minute=minute,
            day_of_week=days
        )

    @classmethod
    def hourly(cls, minute: int = 0) -> 'Schedule':
        """Create hourly schedule at specific minute."""
        return cls(schedule_type=ScheduleType.HOURLY, minute=minute)

    @classmethod
    def cron(cls, expression: str) -> 'Schedule':
        """Create cron schedule from expression."""
        return cls(schedule_type=ScheduleType.CRON, cron_expression=expression)

    @classmethod
    def once(cls, run_at: datetime) -> 'Schedule':
        """Create one-time schedule."""
        return cls(schedule_type=ScheduleType.ONCE, run_at=run_at)

    @classmethod
    def weekdays(cls, hour: int = 9, minute: int = 0) -> 'Schedule':
        """Create weekday schedule (Mon-Fri)."""
        return cls.daily(hour=hour, minute=minute, days=[0, 1, 2, 3, 4])

    def get_next_run(self, after: Optional[datetime] = None) -> Optional[datetime]:
        """Calculate next run time after given datetime."""
        now = after or datetime.now()

        if self.schedule_type == ScheduleType.INTERVAL:
            return now + timedelta(seconds=self.interval_seconds)

        elif self.schedule_type == ScheduleType.HOURLY:
            next_run = now.replace(minute=self.minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(hours=1)
            return next_run

        elif self.schedule_type == ScheduleType.DAILY:
            next_run = now.replace(hour=self.hour, minute=self.minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

            # Check day of week constraint
            if self.day_of_week:
                while next_run.weekday() not in self.day_of_week:
                    next_run += timedelta(days=1)

            return next_run

        elif self.schedule_type == ScheduleType.ONCE:
            if self.run_at and self.run_at > now:
                return self.run_at
            return None

        elif self.schedule_type == ScheduleType.CRON:
            return self._parse_cron_next(now)

        return None

    def _parse_cron_next(self, after: datetime) -> datetime:
        """Parse cron expression and get next run time (simplified)."""
        # Simplified cron parser: minute hour day month weekday
        parts = self.cron_expression.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {self.cron_expression}")

        minute_expr, hour_expr, day_expr, month_expr, weekday_expr = parts

        # Parse minute and hour
        minute = int(minute_expr) if minute_expr != '*' else None
        hour = int(hour_expr) if hour_expr != '*' else None

        # Start from next minute
        next_run = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Find next matching time (simplified)
        for _ in range(1440):  # Max 24 hours of minutes
            matches = True

            if minute is not None and next_run.minute != minute:
                matches = False
            if hour is not None and next_run.hour != hour:
                matches = False

            # Parse weekday constraint
            if weekday_expr != '*':
                if '-' in weekday_expr:
                    start, end = map(int, weekday_expr.split('-'))
                    if not (start <= next_run.weekday() <= end):
                        matches = False
                elif next_run.weekday() != int(weekday_expr):
                    matches = False

            if matches:
                return next_run

            next_run += timedelta(minutes=1)

        return after + timedelta(hours=24)


@dataclass
class JobExecution:
    """Record of a job execution."""
    job_name: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    status: JobStatus = JobStatus.RUNNING
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0


@dataclass
class Job:
    """
    Scheduled job definition.

    Example:
    ```python
    job = Job(
        name="sync_data",
        func=sync_function,
        schedule=Schedule.interval(minutes=30),
        args=["arg1"],
        kwargs={"key": "value"},
        max_retries=3,
        timeout_seconds=300
    )
    ```
    """
    name: str
    func: Callable
    schedule: Schedule
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    max_retries: int = 0
    retry_delay_seconds: int = 60
    timeout_seconds: Optional[int] = None
    depends_on: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    description: str = ""

    # Runtime state
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_status: JobStatus = JobStatus.PENDING
    run_count: int = 0
    error_count: int = 0
    consecutive_failures: int = 0


class JobScheduler:
    """
    Job scheduler with dependency tracking and monitoring.

    Features:
    - Interval, daily, hourly, and cron scheduling
    - Job dependencies
    - Retry with backoff
    - Execution history
    - Concurrent job limits

    Example:
    ```python
    scheduler = JobScheduler(max_concurrent=5)

    # Add jobs
    scheduler.add_job(Job(
        name="fetch_data",
        func=fetch_function,
        schedule=Schedule.interval(minutes=15)
    ))

    scheduler.add_job(Job(
        name="process_data",
        func=process_function,
        schedule=Schedule.interval(minutes=15),
        depends_on=["fetch_data"]  # Runs after fetch_data
    ))

    # Start
    scheduler.start()

    # Check status
    print(scheduler.get_stats())

    # Stop
    scheduler.stop()
    ```
    """

    def __init__(
        self,
        max_concurrent: int = 5,
        history_limit: int = 1000,
        log_file: Optional[str] = None
    ):
        """
        Initialize scheduler.

        Args:
            max_concurrent: Max concurrent jobs
            history_limit: Max execution history entries
            log_file: Optional log file path
        """
        self.max_concurrent = max_concurrent
        self.history_limit = history_limit

        self._jobs: Dict[str, Job] = {}
        self._history: List[JobExecution] = []
        self._running: Set[str] = set()
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None

        # Logging
        self._logger = logging.getLogger("JobScheduler")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def add_job(self, job: Job) -> 'JobScheduler':
        """
        Add a job to the scheduler.

        Args:
            job: Job to add

        Returns:
            Self for chaining
        """
        with self._lock:
            job.next_run = job.schedule.get_next_run()
            self._jobs[job.name] = job
            self._logger.info(f"Added job: {job.name}, next run: {job.next_run}")

        return self

    def remove_job(self, name: str) -> bool:
        """Remove a job from the scheduler."""
        with self._lock:
            if name in self._jobs:
                del self._jobs[name]
                self._logger.info(f"Removed job: {name}")
                return True
            return False

    def get_job(self, name: str) -> Optional[Job]:
        """Get a job by name."""
        return self._jobs.get(name)

    def enable_job(self, name: str) -> bool:
        """Enable a job."""
        job = self._jobs.get(name)
        if job:
            job.enabled = True
            return True
        return False

    def disable_job(self, name: str) -> bool:
        """Disable a job."""
        job = self._jobs.get(name)
        if job:
            job.enabled = False
            return True
        return False

    def run_now(self, name: str) -> Optional[JobExecution]:
        """
        Run a job immediately (ignores schedule).

        Args:
            name: Job name

        Returns:
            JobExecution result or None
        """
        job = self._jobs.get(name)
        if not job:
            return None

        return self._execute_job(job)

    def start(self, blocking: bool = False):
        """
        Start the scheduler.

        Args:
            blocking: If True, blocks until stop() is called
        """
        self._stop_event.clear()
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._scheduler_thread.start()
        self._logger.info("Scheduler started")

        if blocking:
            try:
                while not self._stop_event.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                self.stop()

    def stop(self, wait: bool = True, timeout: float = 30):
        """
        Stop the scheduler.

        Args:
            wait: Wait for running jobs to complete
            timeout: Max wait time
        """
        self._stop_event.set()
        self._logger.info("Scheduler stopping...")

        if wait and self._scheduler_thread:
            self._scheduler_thread.join(timeout=timeout)

        self._logger.info("Scheduler stopped")

    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return not self._stop_event.is_set()

    def _scheduler_loop(self):
        """Main scheduler loop."""
        while not self._stop_event.is_set():
            now = datetime.now()

            # Find jobs ready to run
            ready_jobs = []
            with self._lock:
                for job in self._jobs.values():
                    if not job.enabled:
                        continue
                    if job.name in self._running:
                        continue
                    if job.next_run and job.next_run <= now:
                        # Check dependencies
                        if self._check_dependencies(job):
                            ready_jobs.append(job)

            # Run jobs (up to max concurrent)
            available_slots = self.max_concurrent - len(self._running)
            for job in ready_jobs[:available_slots]:
                self._run_job_async(job)

            # Sleep until next check
            time.sleep(1)

    def _check_dependencies(self, job: Job) -> bool:
        """Check if job dependencies are satisfied."""
        for dep_name in job.depends_on:
            dep_job = self._jobs.get(dep_name)
            if not dep_job:
                continue

            # Dependency must have completed successfully recently
            if dep_job.last_status != JobStatus.COMPLETED:
                return False

            # Dependency should have run after this job's last run
            if job.last_run and dep_job.last_run:
                if dep_job.last_run < job.last_run:
                    return False

        return True

    def _run_job_async(self, job: Job):
        """Run job in a separate thread."""
        with self._lock:
            self._running.add(job.name)

        thread = threading.Thread(target=self._execute_job, args=(job,), daemon=True)
        thread.start()

    def _execute_job(self, job: Job) -> JobExecution:
        """Execute a job with retry logic."""
        execution = JobExecution(
            job_name=job.name,
            started_at=datetime.now()
        )

        self._logger.info(f"Starting job: {job.name}")

        retries = 0
        while retries <= job.max_retries:
            try:
                # Execute function
                result = job.func(*job.args, **job.kwargs)

                # Success
                execution.status = JobStatus.COMPLETED
                execution.result = result
                job.last_status = JobStatus.COMPLETED
                job.consecutive_failures = 0

                self._logger.info(f"Job completed: {job.name}")
                break

            except Exception as e:
                retries += 1
                error_msg = str(e)

                if retries <= job.max_retries:
                    self._logger.warning(f"Job {job.name} failed (attempt {retries}), retrying: {error_msg}")
                    time.sleep(job.retry_delay_seconds * retries)  # Exponential backoff
                else:
                    execution.status = JobStatus.FAILED
                    execution.error = error_msg
                    job.last_status = JobStatus.FAILED
                    job.error_count += 1
                    job.consecutive_failures += 1

                    self._logger.error(f"Job failed: {job.name} - {error_msg}")

        # Update execution record
        execution.ended_at = datetime.now()
        execution.duration_ms = (execution.ended_at - execution.started_at).total_seconds() * 1000

        # Update job state
        with self._lock:
            job.last_run = execution.started_at
            job.run_count += 1
            job.next_run = job.schedule.get_next_run(execution.ended_at)
            self._running.discard(job.name)

            # Add to history
            self._history.append(execution)
            if len(self._history) > self.history_limit:
                self._history = self._history[-self.history_limit:]

        return execution

    def get_history(
        self,
        job_name: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 100
    ) -> List[JobExecution]:
        """
        Get execution history.

        Args:
            job_name: Filter by job name
            status: Filter by status
            limit: Max results

        Returns:
            List of executions
        """
        with self._lock:
            history = self._history.copy()

        if job_name:
            history = [e for e in history if e.job_name == job_name]
        if status:
            history = [e for e in history if e.status == status]

        return history[-limit:]

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        with self._lock:
            jobs_by_status = {}
            for status in JobStatus:
                count = len([j for j in self._jobs.values() if j.last_status == status])
                if count > 0:
                    jobs_by_status[status.value] = count

            return {
                "is_running": self.is_running(),
                "total_jobs": len(self._jobs),
                "running_jobs": len(self._running),
                "jobs_by_status": jobs_by_status,
                "total_executions": len(self._history),
                "failed_executions": len([e for e in self._history if e.status == JobStatus.FAILED]),
                "running_job_names": list(self._running)
            }

    def get_next_jobs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get next scheduled jobs."""
        with self._lock:
            jobs = [
                {"name": j.name, "next_run": j.next_run, "enabled": j.enabled}
                for j in self._jobs.values()
                if j.next_run
            ]

        jobs.sort(key=lambda x: x["next_run"])
        return jobs[:limit]


class SimpleScheduler:
    """
    Simplified scheduler for single-threaded use.

    Useful for scripts that need basic scheduling without threads.

    Example:
    ```python
    scheduler = SimpleScheduler()

    @scheduler.every(minutes=5)
    def check_status():
        print("Checking...")

    @scheduler.daily(hour=8)
    def morning_report():
        print("Morning report")

    # Run once (checks and executes due jobs)
    scheduler.tick()

    # Or run continuously
    scheduler.run()
    ```
    """

    def __init__(self):
        self._jobs: Dict[str, Job] = {}
        self._stop = False

    def every(
        self,
        seconds: int = 0,
        minutes: int = 0,
        hours: int = 0
    ) -> Callable:
        """Decorator for interval scheduling."""
        def decorator(func: Callable) -> Callable:
            job = Job(
                name=func.__name__,
                func=func,
                schedule=Schedule.interval(seconds=seconds, minutes=minutes, hours=hours)
            )
            job.next_run = job.schedule.get_next_run()
            self._jobs[job.name] = job
            return func
        return decorator

    def daily(self, hour: int = 0, minute: int = 0) -> Callable:
        """Decorator for daily scheduling."""
        def decorator(func: Callable) -> Callable:
            job = Job(
                name=func.__name__,
                func=func,
                schedule=Schedule.daily(hour=hour, minute=minute)
            )
            job.next_run = job.schedule.get_next_run()
            self._jobs[job.name] = job
            return func
        return decorator

    def hourly(self, minute: int = 0) -> Callable:
        """Decorator for hourly scheduling."""
        def decorator(func: Callable) -> Callable:
            job = Job(
                name=func.__name__,
                func=func,
                schedule=Schedule.hourly(minute=minute)
            )
            job.next_run = job.schedule.get_next_run()
            self._jobs[job.name] = job
            return func
        return decorator

    def tick(self) -> List[str]:
        """
        Check and run due jobs.

        Returns:
            List of job names that were run
        """
        now = datetime.now()
        ran = []

        for job in self._jobs.values():
            if job.next_run and job.next_run <= now:
                try:
                    job.func(*job.args, **job.kwargs)
                    job.last_status = JobStatus.COMPLETED
                except Exception:
                    job.last_status = JobStatus.FAILED

                job.last_run = now
                job.next_run = job.schedule.get_next_run(now)
                ran.append(job.name)

        return ran

    def run(self, interval: float = 1.0):
        """Run scheduler continuously."""
        self._stop = False
        while not self._stop:
            self.tick()
            time.sleep(interval)

    def stop(self):
        """Stop the scheduler."""
        self._stop = True


# Factory functions

def create_simple_scheduler() -> JobScheduler:
    """Create a basic job scheduler."""
    return JobScheduler(max_concurrent=5)


def create_monitored_scheduler(log_file: str) -> JobScheduler:
    """
    Create a scheduler with logging.

    Args:
        log_file: Path to log file

    Returns:
        Configured JobScheduler
    """
    return JobScheduler(
        max_concurrent=5,
        history_limit=1000,
        log_file=log_file
    )


def create_sequential_scheduler() -> JobScheduler:
    """Create a scheduler that runs one job at a time."""
    return JobScheduler(max_concurrent=1)
