"""Outgoing email for account links (verify email, reset password).

Two senders with the same interface:
- ConsoleMailer (default, development): prints the message to the server console, so links can be
  followed on the laptop without a mail account. Keeps the last messages in memory for tests.
- SmtpMailer (when HL_SMTP_HOST is set): sends over SMTP with STARTTLS. The SMTP password comes from
  the environment only and is never logged.
Sending runs in a background thread so a slow mail server does not delay the page, and a failure is
logged without revealing to the visitor whether the address has an account.
"""
import logging
import smtplib
import threading
from email.message import EmailMessage

log = logging.getLogger("web.mailer")


class ConsoleMailer:
    def __init__(self):
        self.sent: list[dict] = []

    def send(self, to: str, subject: str, body: str) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})
        del self.sent[:-20]
        print(f"\n--- email (console, not sent) ---\nTo: {to}\nSubject: {subject}\n\n{body}\n--- end ---\n", flush=True)


class SmtpMailer:
    def __init__(self, host: str, port: int, user: str, password: str, sender: str):
        self.host, self.port, self.user, self.password, self.sender = host, port, user, password, sender

    def _send(self, msg: EmailMessage) -> None:
        try:
            with smtplib.SMTP(self.host, self.port, timeout=20) as s:
                s.starttls()
                if self.user:
                    s.login(self.user, self.password)
                s.send_message(msg)
        except (OSError, smtplib.SMTPException) as e:
            log.error("email to %s failed: %s", msg["To"], type(e).__name__)

    def send(self, to: str, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["From"], msg["To"], msg["Subject"] = self.sender, to, subject
        msg.set_content(body)
        threading.Thread(target=self._send, args=(msg,), daemon=True).start()


def make_mailer(smtp: dict):
    return SmtpMailer(**smtp) if smtp else ConsoleMailer()


SIGNATURE = ("\n\n-- \nHyperlocal forecasts for Nakuru and Busia (research prototype, Strathmore University).\n"
             "Not an official forecast: the Kenya Meteorological Department is the authority.\n"
             "If you did not ask for this email, you can ignore it.")


def verify_message(name: str, link: str) -> tuple[str, str]:
    return ("Confirm your email", f"Hello {name},\n\nWelcome! Confirm your email address by opening this link "
            f"(valid for 3 days):\n\n{link}{SIGNATURE}")


def reset_message(name: str, link: str) -> tuple[str, str]:
    return ("Reset your password", f"Hello {name},\n\nSomeone (hopefully you) asked to reset the password for your "
            f"account. Open this link within 30 minutes to choose a new one:\n\n{link}\n\nThe link works once."
            f"{SIGNATURE}")
