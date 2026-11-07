"""Manage dashboard accounts (invite-only: there is no sign-up page).

    python -m web.users add <username> --role admin|viewer    # asks for the password twice
    python -m web.users list
    python -m web.users reset-password <username>
    python -m web.users disable <username>     /   enable <username>
    python -m web.users remove <username>
    python -m web.users audit                  # recent logins, failures and lockouts

Passwords are typed at a hidden prompt (getpass) and never appear in the shell history.
The database is the one the server uses (HL_AUTH_DB, default data/auth/users.db).
"""
import argparse
import getpass
import sys
from datetime import datetime

from web.auth import MIN_PASSWORD, AuthStore, PolicyError
from web.settings import settings


def ask_password() -> str:
    for _ in range(3):
        p1 = getpass.getpass(f"Password (at least {MIN_PASSWORD} characters): ")
        p2 = getpass.getpass("Repeat password: ")
        if p1 == p2:
            return p1
        print("The two passwords differ, try again.")
    raise SystemExit("No password set.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Manage dashboard accounts.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("username")
    a.add_argument("--role", choices=["admin", "viewer"], default="viewer")
    for name in ("reset-password", "disable", "enable", "remove"):
        sub.add_parser(name).add_argument("username")
    sub.add_parser("list")
    sub.add_parser("audit")
    args = parser.parse_args(argv)

    store = AuthStore(settings().auth_db)
    try:
        if args.cmd == "add":
            store.add_user(args.username, ask_password(), args.role)
            print(f"added {args.username} ({args.role})")
        elif args.cmd == "reset-password":
            store.set_password(args.username, ask_password())
            print(f"password changed for {args.username}; their sessions were signed out")
        elif args.cmd in ("disable", "enable"):
            store.set_disabled(args.username, args.cmd == "disable")
            print(f"{args.cmd}d {args.username}")
        elif args.cmd == "remove":
            store.remove_user(args.username)
            print(f"removed {args.username}")
        elif args.cmd == "list":
            users = store.list_users()
            if not users:
                print("no users yet: python -m web.users add <name> --role admin")
            for u in users:
                flags = ", ".join(f for f, on in (("disabled", u["disabled"]), ("locked", u["locked"])) if on)
                print(f"  {u['username']:<20} {u['role']:<7} {flags}")
        elif args.cmd == "audit":
            for e in reversed(store.recent_audit(30)):
                print(f"  {datetime.fromtimestamp(e['at']):%Y-%m-%d %H:%M:%S}  {e['event']:<15} {e['username'] or '':<20} {e['ip'] or ''}")
    except PolicyError as e:
        sys.exit(f"error: {e}")
    finally:
        store.close()


if __name__ == "__main__":
    main()
