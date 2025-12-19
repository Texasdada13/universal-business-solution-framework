"""
Universal Task Scheduling Patterns
Run tasks automatically at specified intervals

Based on: NYPD Procurement Tracker
"""

import schedule
import time
import argparse
from datetime import datetime


class UniversalScheduler:
    """
    Generic task scheduler with configurable intervals

    Usage:
        scheduler = UniversalScheduler(interval_minutes=60)
        scheduler.set_task(my_function)
        scheduler.start()  # Runs forever
    """

    def __init__(self, interval_minutes=60, task_name="Scheduled Task"):
        """
        Initialize scheduler

        Args:
            interval_minutes (int): How often to run the task (in minutes)
            task_name (str): Name of the task for logging
        """
        self.interval_minutes = interval_minutes
        self.task_name = task_name
        self.task_function = None
        self.run_count = 0

    def set_task(self, task_function):
        """
        Set the function to run on schedule

        Args:
            task_function (callable): Function to run (no arguments)
        """
        self.task_function = task_function

    def run_task(self):
        """
        Run the scheduled task with error handling and logging
        """
        if not self.task_function:
            print("[ERROR] No task function set")
            return

        self.run_count += 1

        print("=" * 70)
        print(f"{self.task_name} - Run #{self.run_count}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)

        try:
            # Run the task
            self.task_function()

            print("-" * 70)
            print(f"[OK] {self.task_name} completed successfully")
            print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Next run in {self.interval_minutes} minutes")
            print("=" * 70)

        except Exception as e:
            print("-" * 70)
            print(f"[ERROR] {self.task_name} failed: {e}")
            print(f"Will retry in {self.interval_minutes} minutes")
            print("=" * 70)

    def start(self, run_immediately=True):
        """
        Start the scheduler

        Args:
            run_immediately (bool): Run task immediately before starting schedule
        """
        print("=" * 70)
        print(f"STARTING SCHEDULER: {self.task_name}")
        print("=" * 70)
        print(f"Interval: Every {self.interval_minutes} minutes")
        print(f"Press Ctrl+C to stop")
        print("=" * 70)
        print()

        # Schedule the task
        schedule.every(self.interval_minutes).minutes.do(self.run_task)

        # Run immediately if requested
        if run_immediately:
            self.run_task()

        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("Scheduler stopped by user")
            print(f"Total runs completed: {self.run_count}")
            print("=" * 70)

    def run_once(self):
        """
        Run the task once and exit (useful for testing)
        """
        print(f"Running {self.task_name} once...")
        self.run_task()


class MultiTaskScheduler:
    """
    Schedule multiple tasks with different intervals

    Usage:
        scheduler = MultiTaskScheduler()
        scheduler.add_task("Scrape Data", scrape_function, interval_minutes=30)
        scheduler.add_task("Generate Report", report_function, interval_minutes=1440)  # Daily
        scheduler.start()
    """

    def __init__(self):
        """Initialize multi-task scheduler"""
        self.tasks = []

    def add_task(self, name, task_function, interval_minutes):
        """
        Add a task to the scheduler

        Args:
            name (str): Task name
            task_function (callable): Function to run
            interval_minutes (int): Interval in minutes
        """
        self.tasks.append({
            'name': name,
            'function': task_function,
            'interval': interval_minutes,
            'run_count': 0
        })
        print(f"[OK] Added task: {name} (every {interval_minutes} minutes)")

    def create_task_wrapper(self, task):
        """
        Create a wrapper function for a task with logging

        Args:
            task (dict): Task configuration

        Returns:
            callable: Wrapped task function
        """
        def wrapper():
            task['run_count'] += 1
            print("=" * 70)
            print(f"{task['name']} - Run #{task['run_count']}")
            print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 70)

            try:
                task['function']()
                print(f"[OK] {task['name']} completed")
            except Exception as e:
                print(f"[ERROR] {task['name']} failed: {e}")

            print("=" * 70)
            print()

        return wrapper

    def start(self):
        """Start all scheduled tasks"""
        print("=" * 70)
        print("MULTI-TASK SCHEDULER")
        print("=" * 70)
        print(f"Tasks scheduled: {len(self.tasks)}")
        for task in self.tasks:
            print(f"  - {task['name']}: every {task['interval']} minutes")
        print()
        print("Press Ctrl+C to stop")
        print("=" * 70)
        print()

        # Schedule all tasks
        for task in self.tasks:
            wrapper = self.create_task_wrapper(task)
            schedule.every(task['interval']).minutes.do(wrapper)

        # Run all tasks immediately
        for task in self.tasks:
            wrapper = self.create_task_wrapper(task)
            wrapper()

        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("Scheduler stopped by user")
            print("Task Summary:")
            for task in self.tasks:
                print(f"  {task['name']}: {task['run_count']} runs")
            print("=" * 70)


# Helper function for command-line arguments
def create_cli_scheduler(task_function, default_interval=60, task_name="Scheduled Task"):
    """
    Create a scheduler with command-line argument support

    Args:
        task_function (callable): Function to run
        default_interval (int): Default interval in minutes
        task_name (str): Task name

    Returns:
        UniversalScheduler: Configured scheduler
    """
    parser = argparse.ArgumentParser(description=f'Run {task_name} on schedule')
    parser.add_argument('--interval', type=int, default=default_interval,
                        help=f'Check interval in minutes (default: {default_interval})')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (for testing)')
    args = parser.parse_args()

    scheduler = UniversalScheduler(interval_minutes=args.interval, task_name=task_name)
    scheduler.set_task(task_function)

    if args.once:
        scheduler.run_once()
    else:
        scheduler.start()

    return scheduler


# Example usage
if __name__ == "__main__":
    # Example 1: Simple scheduled task
    def my_task():
        """Example task"""
        print("Executing my task...")
        # Your task logic here
        time.sleep(2)  # Simulate work
        print("Task completed!")

    scheduler = UniversalScheduler(interval_minutes=5, task_name="My Example Task")
    scheduler.set_task(my_task)
    # scheduler.start()  # Uncomment to run

    # Example 2: Multiple tasks
    def scrape_data():
        print("Scraping data...")

    def generate_report():
        print("Generating report...")

    multi_scheduler = MultiTaskScheduler()
    multi_scheduler.add_task("Scrape Data", scrape_data, interval_minutes=30)
    multi_scheduler.add_task("Generate Report", generate_report, interval_minutes=1440)
    # multi_scheduler.start()  # Uncomment to run

    # Example 3: CLI scheduler (use in main script)
    # if __name__ == "__main__":
    #     create_cli_scheduler(my_task, default_interval=60, task_name="Data Processor")
