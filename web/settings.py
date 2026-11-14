"""Dashboard settings from environment variables (defaults are for running on the laptop).

    HL_HOST            address to listen on (default 127.0.0.1; use 0.0.0.0 only behind HTTPS when hosting)
    HL_PORT            port (default 8765)
    HL_AUTH_DB         SQLite file with users, sessions and saved places (default data/auth/users.db)
    HL_SECURE_COOKIES  1 when the site is served over HTTPS: cookies get the Secure flag and the
                       __Host- prefix (default 0, because the laptop serves plain http://127.0.0.1)
    HL_SIGNUP          1 = anyone can create an account on /signup (default); 0 = invite-only (accounts
                       made with python -m web.users)
    HL_BASE_URL        address used in emailed links, e.g. https://forecast.example.org (default
                       http://<host>:<port>). Never taken from the request, so a forged Host header
                       cannot redirect a password-reset link.
    HL_SMTP_HOST, HL_SMTP_PORT (587), HL_SMTP_USER, HL_SMTP_PASSWORD, HL_MAIL_FROM
                       outgoing mail (STARTTLS). Without HL_SMTP_HOST, emails are printed to the server
                       console instead (development).
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    auth_db: Path
    secure_cookies: bool
    signup_open: bool = True
    base_url: str = ""
    smtp: dict = field(default_factory=dict)

    @property
    def cookie_name(self) -> str:
        return "__Host-hl_session" if self.secure_cookies else "hl_session"

    @property
    def link_base(self) -> str:
        return (self.base_url or f"http://{self.host}:{self.port}").rstrip("/")


def settings() -> Settings:
    db = Path(os.environ.get("HL_AUTH_DB", ROOT / "data" / "auth" / "users.db"))
    smtp = {}
    if os.environ.get("HL_SMTP_HOST"):
        smtp = {"host": os.environ["HL_SMTP_HOST"], "port": int(os.environ.get("HL_SMTP_PORT", "587")),
                "user": os.environ.get("HL_SMTP_USER", ""), "password": os.environ.get("HL_SMTP_PASSWORD", ""),
                "sender": os.environ.get("HL_MAIL_FROM", os.environ.get("HL_SMTP_USER", ""))}
    return Settings(host=os.environ.get("HL_HOST", "127.0.0.1"), port=int(os.environ.get("HL_PORT", "8765")),
                    auth_db=db if db.is_absolute() else ROOT / db,
                    secure_cookies=os.environ.get("HL_SECURE_COOKIES", "0") == "1",
                    signup_open=os.environ.get("HL_SIGNUP", "1") == "1",
                    base_url=os.environ.get("HL_BASE_URL", ""), smtp=smtp)
