from __future__ import annotations

import json
import logging
import urllib.request
from urllib.error import URLError

from PyQt6.QtCore import QThread, pyqtSignal

from .version import __version__, API_URL

logger = logging.getLogger(__name__)


def _parse_tag(tag: str) -> tuple[int, ...]:
    """'v1.2.3' or '1.2.3' -> (1, 2, 3)"""
    return tuple(int(x) for x in tag.lstrip("v").split(".") if x.isdigit())


class UpdateChecker(QThread):
    """Fetches the latest GitHub release in a background thread.

    Emits update_available(latest_version, release_url) if a newer
    version is found. Silently does nothing on network errors.
    """

    update_available = pyqtSignal(str, str)  # (tag, html_url)

    def run(self) -> None:
        try:
            req = urllib.request.Request(
                API_URL,
                headers={"User-Agent": f"RawriisSTT/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode())

            tag: str = data.get("tag_name", "")
            url: str = data.get("html_url", "")
            if not tag:
                return

            if _parse_tag(tag) > _parse_tag(__version__):
                self.update_available.emit(tag, url)

        except (URLError, OSError, ValueError, KeyError):
            pass  # no network, rate-limited, bad JSON — silently ignore
