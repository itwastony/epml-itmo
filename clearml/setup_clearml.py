#!/usr/bin/env python3
"""
ClearML Setup Script

This script helps to configure ClearML for the project.
It can be used to:
1. Initialize ClearML configuration
2. Test connection to ClearML server
3. Create default project and experiments

Usage:
    python clearml/setup_clearml.py --init       # Initialize configuration
    python clearml/setup_clearml.py --test       # Test connection
    python clearml/setup_clearml.py --create-project  # Create project
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def init_clearml_config() -> None:
    """Initialize ClearML configuration interactively."""
    print("=" * 60)
    print("ClearML Configuration Setup")
    print("=" * 60)
    print()
    print("This will help you configure ClearML for the project.")
    print()
    print("Prerequisites:")
    print("1. ClearML Server running (docker-compose up -d)")
    print("2. Access to ClearML Web UI (http://localhost:8080)")
    print()
    print("Steps:")
    print("1. Open ClearML Web UI")
    print("2. Go to Settings -> Workspace -> Create new credentials")
    print("3. Copy the credentials")
    print()

    try:
        import clearml  # noqa: F401

        print("\nTo configure ClearML, run:")
        print("  clearml-init")
        print()
        print("Or set environment variables:")
        print("  export CLEARML_API_HOST=http://localhost:8008")
        print("  export CLEARML_WEB_HOST=http://localhost:8080")
        print("  export CLEARML_FILES_HOST=http://localhost:8081")
        print("  export CLEARML_API_ACCESS_KEY=<your_access_key>")
        print("  export CLEARML_API_SECRET_KEY=<your_secret_key>")

    except ImportError:
        print("ERROR: ClearML not installed. Run: poetry add clearml")
        sys.exit(1)


def test_clearml_connection() -> bool:
    """Test connection to ClearML server."""
    print("=" * 60)
    print("Testing ClearML Connection")
    print("=" * 60)
    print()

    try:
        from typing import Any

        from clearml import Task

        # Try to create a test task
        task: Any = Task.init(
            project_name="EPML-ITMO/Test",
            task_name="Connection Test",
            task_type=Task.TaskTypes.testing,
            reuse_last_task_id=False,
        )

        print("✓ Successfully connected to ClearML!")
        print(f"  Task ID: {task.id}")
        print(f"  Project: {task.get_project_name()}")

        # Log a test metric
        task.get_logger().report_scalar(
            title="test", series="connection", value=1, iteration=0
        )

        task.close()
        print("\n✓ Connection test passed!")
        return True

    except Exception as e:
        print(f"\n✗ Connection test failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if ClearML server is running")
        print("2. Verify credentials in ~/clearml.conf")
        print("3. Check network connectivity")
        return False


def create_project() -> None:
    """Create default project structure in ClearML."""
    print("=" * 60)
    print("Creating ClearML Project")
    print("=" * 60)
    print()

    try:
        from typing import Any

        from clearml import Task

        # Create main project with a setup task
        task: Any = Task.init(
            project_name="EPML-ITMO/Wine-Quality",
            task_name="Project Setup",
            task_type=Task.TaskTypes.custom,
            reuse_last_task_id=False,
        )

        # Add project description
        task.set_comment(
            """
Wine Quality Classification Project
=====================================

This project uses ClearML for MLOps workflow management.

Models:
- Random Forest
- Gradient Boosting
- Logistic Regression
- SVM
- Decision Tree
- KNN

Dataset:
- Wine Quality (Red Wine) from UCI ML Repository

Experiments Structure:
- EPML-ITMO/Wine-Quality/Experiments - Training experiments
- EPML-ITMO/Wine-Quality/Models - Model registry
- EPML-ITMO/Wine-Quality/Pipelines - ML Pipelines
            """
        )

        # Add tags
        task.add_tags(["setup", "project-init"])

        print("✓ Created project: EPML-ITMO/Wine-Quality")
        print(f"  Task ID: {task.id}")

        task.close()
        print("\n✓ Project created successfully!")

    except Exception as e:
        print(f"\n✗ Failed to create project: {e}")
        sys.exit(1)


def show_status() -> None:
    """Show current ClearML configuration status."""
    print("=" * 60)
    print("ClearML Configuration Status")
    print("=" * 60)
    print()

    # Check for config file
    config_paths = [
        Path.home() / "clearml.conf",
        Path.home() / ".clearml" / "clearml.conf",
        Path("/etc/clearml.conf"),
    ]

    config_found = False
    for path in config_paths:
        if path.exists():
            print(f"✓ Config file found: {path}")
            config_found = True
            break

    if not config_found:
        print("✗ Config file not found")
        print("  Expected locations:")
        for path in config_paths:
            print(f"    - {path}")

    # Check environment variables
    env_vars = [
        "CLEARML_API_HOST",
        "CLEARML_WEB_HOST",
        "CLEARML_FILES_HOST",
        "CLEARML_API_ACCESS_KEY",
        "CLEARML_API_SECRET_KEY",
    ]

    print("\nEnvironment Variables:")
    for var in env_vars:
        value = os.environ.get(var, "")
        if value:
            # Mask sensitive values
            if "KEY" in var:
                display_value = value[:4] + "..." if len(value) > 4 else "***"
            else:
                display_value = value
            print(f"  ✓ {var}: {display_value}")
        else:
            print(f"  ✗ {var}: not set")

    # Try to import and check configuration
    print("\nClearML Package:")
    try:
        import clearml

        version = getattr(clearml, "__version__", "unknown")
        print(f"  ✓ Version: {version}")
    except ImportError:
        print("  ✗ Not installed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="ClearML Setup Utility")
    parser.add_argument("--init", action="store_true", help="Initialize configuration")
    parser.add_argument("--test", action="store_true", help="Test connection")
    parser.add_argument(
        "--create-project", action="store_true", help="Create project structure"
    )
    parser.add_argument(
        "--status", action="store_true", help="Show configuration status"
    )

    args = parser.parse_args()

    if args.init:
        init_clearml_config()
    elif args.test:
        success = test_clearml_connection()
        sys.exit(0 if success else 1)
    elif args.create_project:
        create_project()
    elif args.status:
        show_status()
    else:
        # Default: show status
        show_status()
        print("\nUsage:")
        print("  python clearml/setup_clearml.py --init          # Initialize")
        print("  python clearml/setup_clearml.py --test          # Test connection")
        print("  python clearml/setup_clearml.py --create-project # Create project")
        print("  python clearml/setup_clearml.py --status        # Show status")


if __name__ == "__main__":
    main()
