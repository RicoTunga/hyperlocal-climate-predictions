// Show the reason for a failed sign-in (from ?error=...) and focus the first field.
const messages = {
  invalid: "Wrong username or password.",
  locked: "Too many failed attempts. Try again in a few minutes.",
  rate: "Too many sign-in attempts from this connection. Wait a few minutes.",
  expired: "Your session ended. Please sign in again.",
};
const code = new URLSearchParams(location.search).get("error");
const box = document.getElementById("error");
if (code && messages[code]) {
  box.textContent = messages[code];
  box.hidden = false;
}
document.getElementById("username").focus();
