"""Dashboard settings from environment variables (defaults are for running on the laptop).

    HL_HOST            address to listen on (default 127.0.0.1; use 0.0.0.0 only behind HTTPS when hosting)
    HL_PORT            port (default 8765)
    HL_AUTH_DB         SQLite file with users and sessions (default data/auth/users.db)
    HL_SECURE_COOKIES  1 when the site is served over HTTPS: cookies get the Secure flag and the
                       __Host- prefix (default 0, because the laptop serves plain http://127.0.0.1)
"""
import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    auth_db: Path
    secure_cookies: bool

    @property
    def cookie_name(self) -> str:
        return "__Host-hl_session" if self.secure_cookies else "hl_session"


def settings() -> Settings:
    db = Path(os.environ.get("HL_AUTH_DB", ROOT / "data" / "auth" / "users.db"))
    return Settings(host=os.environ.get("HL_HOST", "127.0.0.1"), port=int(os.environ.get("HL_PORT", "8765")),
                    auth_db=db if db.is_absolute() else ROOT / db,
                    secure_cookies=os.environ.get("HL_SECURE_COOKIES", "0") == "1")
