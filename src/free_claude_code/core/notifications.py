"""Native OS task notification dispatcher for Free Claude Code."""

import os
import subprocess
import sys

from loguru import logger


def send_task_notification(title: str, message: str) -> bool:
    """Send a native OS desktop notification when a task completes."""
    if os.environ.get("CI") or os.environ.get("FCC_HEADLESS"):
        return False

    try:
        if sys.platform == "win32":
            ps_script = (
                "[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null; "
                "$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02); "
                "$textNodes = $template.GetElementsByTagName('text'); "
                f"$textNodes.Item(0).AppendChild($template.CreateTextNode('{title}')) > $null; "
                f"$textNodes.Item(1).AppendChild($template.CreateTextNode('{message}')) > $null; "
                "$toast = [Windows.UI.Notifications.ToastNotification]::new($template); "
                "[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('Free Claude Code').Show($toast);"
            )
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-WindowStyle",
                    "Hidden",
                    "-Command",
                    ps_script,
                ],
                check=False,
                capture_output=True,
                timeout=5.0,
            )
            return True
        elif sys.platform == "darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ],
                check=False,
                capture_output=True,
                timeout=3.0,
            )
            return True
        else:
            subprocess.run(
                ["notify-send", title, message],
                check=False,
                capture_output=True,
                timeout=3.0,
            )
            return True
    except Exception as exc:
        logger.debug("Desktop notification dispatch skipped: {}", exc)
        return False
