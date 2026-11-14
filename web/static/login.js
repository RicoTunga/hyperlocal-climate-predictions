// Show the reason for a failed sign-in (from ?error=...) or a confirmation, and focus the first field.
const errors = {
  invalid: "Wrong username, email or password.",
  locked: "Too many failed attempts. Try again in a few minutes.",
  rate: "Too many sign-in attempts from this connection. Wait a few minutes.",
  expired: "Your session ended. Please sign in again.",
  closed: "New accounts are by invitation only on this server. Ask an administrator.",
};
const notes = {
  reset: "Password changed. Sign in with your new password.",
  deleted: "Your account and saved data were deleted.",
};
const q = new URLSearchParams(location.search);
const code = q.get("error");
if (code && errors[code]) {
  const box = document.getElementById("error");
  box.textContent = errors[code];
  box.hidden = false;
}
const note = q.get("verified") === "1" ? "Email confirmed. Sign in to continue."
  : q.get("verified") === "0" ? null : notes[[...q.keys()].find((k) => notes[k])];
if (note) {
  const ok = document.getElementById("ok");
  ok.textContent = note;
  ok.hidden = false;
}
if (q.get("verified") === "0") {
  const box = document.getElementById("error");
  box.textContent = "That confirmation link has expired or was already used. Sign in and ask for a new one from your account.";
  box.hidden = false;
}
fetch("/auth/config").then((r) => r.json()).then((c) => {
  document.getElementById("signup-link").hidden = !c.signup_open;
}).catch(() => {});
document.getElementById("username").focus();
