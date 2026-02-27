"""
Pipeline monitoring and notification system.

Provides execution tracking, timing, and reporting for ML pipelines.
"""

import json
import logging
import smtplib
import time
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class PipelineMonitor:
    """
    Monitor for tracking pipeline execution.

    Tracks timing, success/failure status, and generates reports.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize pipeline monitor.

        Args:
            cfg: Hydra configuration object
        """
        self.cfg = cfg
        self.stages: dict[str, dict[str, Any]] = {}
        self.pipeline_start: float | None = None
        self.pipeline_end: float | None = None
        self.current_stage: str | None = None

    def start_pipeline(self) -> None:
        """Mark pipeline start."""
        self.pipeline_start = time.time()
        logger.info("=" * 60)
        logger.info("PIPELINE STARTED")
        logger.info(f"Model: {self.cfg.model.name}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)

    def end_pipeline(
        self, success: bool = True, error: str | None = None
    ) -> dict[str, Any]:
        """
        Mark pipeline end and generate summary.

        Args:
            success: Whether pipeline completed successfully
            error: Error message if failed

        Returns:
            Pipeline execution report
        """
        self.pipeline_end = time.time()
        duration = self.pipeline_end - (self.pipeline_start or 0)

        report = {
            "success": success,
            "model": self.cfg.model.name,
            "total_duration_seconds": round(duration, 2),
            "timestamp": datetime.now().isoformat(),
            "stages": self.stages,
        }

        if error:
            report["error"] = error

        # Log summary
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED")
        logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")
        logger.info(f"Total duration: {duration:.2f}s")
        logger.info("-" * 40)

        for stage_name, stage_info in self.stages.items():
            status = "✓" if stage_info.get("success") else "✗"
            stage_duration = stage_info.get("duration", 0)
            logger.info(f"  {status} {stage_name}: {stage_duration:.2f}s")

        if error:
            logger.error(f"Error: {error}")

        logger.info("=" * 60)

        # Send notification
        self._send_notification(report)

        return report

    def start_stage(self, stage_name: str) -> None:
        """
        Mark stage start.

        Args:
            stage_name: Name of the stage
        """
        self.current_stage = stage_name
        self.stages[stage_name] = {
            "start_time": time.time(),
            "status": "running",
        }
        logger.info(f"[STAGE] Starting: {stage_name}")

    def end_stage(
        self, stage_name: str, success: bool = True, error: str | None = None
    ) -> None:
        """
        Mark stage end.

        Args:
            stage_name: Name of the stage
            success: Whether stage completed successfully
            error: Error message if failed
        """
        if stage_name not in self.stages:
            logger.warning(f"Stage {stage_name} was not started")
            return

        end_time = time.time()
        start_time = self.stages[stage_name]["start_time"]
        duration = end_time - start_time

        self.stages[stage_name].update(
            {
                "end_time": end_time,
                "duration": round(duration, 2),
                "success": success,
                "status": "completed" if success else "failed",
            }
        )

        if error:
            self.stages[stage_name]["error"] = error

        status_msg = "✓ Completed" if success else "✗ Failed"
        logger.info(f"[STAGE] {status_msg}: {stage_name} ({duration:.2f}s)")

        self.current_stage = None

    def save_report(self, output_path: Path | None = None) -> Path:
        """
        Save execution report to file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = Path(self.cfg.output_dir) / "pipeline_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model": self.cfg.model.name,
            "experiment": self.cfg.mlflow.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "total_duration": (
                round((self.pipeline_end or 0) - (self.pipeline_start or 0), 2)
            ),
            "stages": self.stages,
            "config": {
                "model_params": dict(self.cfg.model.params),
                "training": dict(self.cfg.training),
                "seed": self.cfg.seed,
            },
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Pipeline report saved to: {output_path}")
        return output_path

    def _send_notification(self, report: dict[str, Any]) -> None:
        """
        Send notification about pipeline completion.

        Currently logs to file. Can be extended for email/Slack notifications.

        Args:
            report: Pipeline execution report
        """
        # Log notification to file
        notification_file = Path(self.cfg.output_dir) / "notifications.log"
        notification_file.parent.mkdir(parents=True, exist_ok=True)

        status = "SUCCESS" if report["success"] else "FAILED"
        message = f"""
================================================================================
PIPELINE NOTIFICATION - {status}
================================================================================
Model: {report['model']}
Timestamp: {report['timestamp']}
Duration: {report['total_duration_seconds']}s

Stages:
"""
        for stage_name, stage_info in report.get("stages", {}).items():
            stage_status = "✓" if stage_info.get("success") else "✗"
            message += (
                f"  {stage_status} {stage_name}: {stage_info.get('duration', 0):.2f}s\n"
            )

        if not report["success"]:
            message += f"\nError: {report.get('error', 'Unknown error')}\n"

        message += "=" * 80 + "\n"

        with open(notification_file, "a") as f:
            f.write(message)

        logger.info(f"Notification logged to: {notification_file}")

        # Console notification
        if report["success"]:
            logger.info("🎉 Pipeline completed successfully!")
        else:
            logger.error("❌ Pipeline failed! Check logs for details.")


class EmailNotifier:
    """
    Email notification handler for pipeline events.

    Note: Requires SMTP configuration. Disabled by default.
    """

    def __init__(
        self,
        smtp_server: str = "localhost",
        smtp_port: int = 587,
        username: str | None = None,
        password: str | None = None,
    ):
        """
        Initialize email notifier.

        Args:
            smtp_server: SMTP server address
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
        """
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password

    def send(
        self,
        to_email: str,
        subject: str,
        body: str,
        from_email: str | None = None,
    ) -> bool:
        """
        Send email notification.

        Args:
            to_email: Recipient email address
            subject: Email subject
            body: Email body
            from_email: Sender email address

        Returns:
            True if sent successfully, False otherwise
        """
        try:
            msg = MIMEText(body)
            msg["Subject"] = subject
            msg["From"] = from_email or self.username or "noreply@localhost"
            msg["To"] = to_email

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.username and self.password:
                    server.starttls()
                    server.login(self.username, self.password)
                server.send_message(msg)

            logger.info(f"Email notification sent to {to_email}")
            return True

        except Exception as e:
            logger.warning(f"Failed to send email notification: {e}")
            return False


def format_pipeline_report(report: dict[str, Any]) -> str:
    """
    Format pipeline report as readable text.

    Args:
        report: Pipeline execution report

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "ML PIPELINE EXECUTION REPORT",
        "=" * 60,
        f"Model: {report.get('model', 'N/A')}",
        f"Status: {'SUCCESS' if report.get('success') else 'FAILED'}",
        f"Timestamp: {report.get('timestamp', 'N/A')}",
        f"Total Duration: {report.get('total_duration_seconds', 0):.2f}s",
        "",
        "-" * 40,
        "STAGES:",
        "-" * 40,
    ]

    for stage_name, stage_info in report.get("stages", {}).items():
        status = "✓" if stage_info.get("success") else "✗"
        duration = stage_info.get("duration", 0)
        lines.append(f"  {status} {stage_name}: {duration:.2f}s")

        if not stage_info.get("success") and stage_info.get("error"):
            lines.append(f"      Error: {stage_info['error']}")

    if not report.get("success") and report.get("error"):
        lines.extend(["", f"Pipeline Error: {report['error']}"])

    lines.append("=" * 60)

    return "\n".join(lines)
