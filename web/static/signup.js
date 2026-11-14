// Sign-up page: checks the form in the browser for quick feedback; the server checks everything again.
const $ = (id) => document.getElementById(id);
const form = $("signup-form"), box = $("error"), btn = $("submit");

function show(msg) {
  box.textContent = msg.charAt(0).toUpperCase() + msg.slice(1);
  box.hidden = false;
  box.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

// a rough strength hint: length and variety (the server enforces the real rules)
function strength(pw) {
  if (!pw) return 0;
  const kinds = [/[a-z]/, /[A-Z]/, /\d/, /[^A-Za-z0-9]/, / /].filter((r) => r.test(pw)).length;
  return Math.min(1, (pw.length / 20) * 0.7 + (kinds / 5) * 0.3);
}
$("password").addEventListener("input", (e) => {
  const s = strength(e.target.value), bar = $("meter-bar");
  bar.style.width = `${Math.round(100 * s)}%`;
  bar.dataset.level = s < 0.45 ? "weak" : s < 0.75 ? "ok" : "strong";
});

// suggest a username from the name, until the user types their own
let touched = false;
$("username").addEventListener("input", () => { touched = true; });
$("display_name").addEventListener("input", (e) => {
  if (!touched) $("username").value = e.target.value.trim().toLowerCase().replace(/\s+/g, ".").replace(/[^a-z0-9_.-]/g, "").slice(0, 32);
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  box.hidden = true;
  const data = Object.fromEntries(new FormData(form));
  data.agree = $("agree").checked;
  if (!data.display_name.trim()) return show("please enter your name");
  if (!/^[A-Za-z0-9_.-]{3,32}$/.test(data.username)) return show("username: 3–32 letters, digits, '.', '_' or '-'");
  if (!$("email").checkValidity()) return show("please enter a valid email address");
  if (data.password.length < 12) return show("password: at least 12 characters");
  if (data.password !== data.password2) return show("the two passwords differ");
  if (!data.agree) return show("please tick the box to confirm you understand this is a research prototype");
  btn.disabled = true; btn.textContent = "Creating your account…";
  try {
    const r = await fetch("/signup", { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(j.error || `sign-up failed (${r.status})`);
    location.href = j.next || "/";
  } catch (err) {
    show(err.message);
    btn.disabled = false; btn.textContent = "Create account";
  }
});
$("display_name").focus();
