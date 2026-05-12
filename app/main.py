"""Application entry point."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from app.config.settings import AppSettings
from app.gui.main_window import MainWindow


def configure_logging() -> None:
    """Configure basic console logging for the skeleton app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> int:
    """Run the desktop application."""
    configure_logging()
    settings = AppSettings.from_env()

    app = QApplication(sys.argv)
    app.setApplicationName(settings.app_name)
    app.setApplicationVersion(settings.app_version)

    window = MainWindow(settings=settings)
    window.show()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
