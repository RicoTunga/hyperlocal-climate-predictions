// "Forgot password" and "choose a new password" pages.
const $ = (id) => document.getElementById(id);
const form = document.querySelector("form"), mode = form.dataset.mode, box = $("error"), btn = $("submit");

function show(msg) { box.textContent = msg.charAt(0).toUpperCase() + msg.slice(1); box.hidden = false; }

async function post(path, data) {
  const r = await fetch(path, { method: "POST", credentials: "same-origin",
    headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `request failed (${r.status})`);
  return j;
}

// the reset token travels in the link; take it out of the address bar so it is not kept in history
const token = new URLSearchParams(location.search).get("token") || "";
if (mode === "reset") {
  history.replaceState(null, "", "/reset");
  if (!token) { show("this page needs the link from your email"); btn.disabled = true; }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  box.hidden = true;
  const label = btn.textContent;
  try {
    if (mode === "forgot") {
      if (!$("email").checkValidity() || !$("email").value) return show("please enter a valid email address");
      btn.disabled = true; btn.textContent = "Sending…";
      const j = await post("/forgot", { email: $("email").value.trim() });
      $("ok").textContent = j.message; $("ok").hidden = false;
      form.hidden = true;
    } else {
      const pw = $("password").value, pw2 = $("password2").value;
      if (pw.length < 12) return show("password: at least 12 characters");
      if (pw !== pw2) return show("the two passwords differ");
      btn.disabled = true; btn.textContent = "Saving…";
      const j = await post("/reset", { token, password: pw, password2: pw2 });
      location.href = j.next;
    }
  } catch (err) {
    show(err.message);
  } finally {
    btn.disabled = false; btn.textContent = label;
  }
});
(mode === "forgot" ? $("email") : $("password")).focus();
