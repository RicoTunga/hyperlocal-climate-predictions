# Hosting the dashboard

The dashboard is ready to host but **has not been deployed yet**, and the Docker image has **not been built
or tested** on the development laptop (Docker is not installed there). Check every step below on the first
deployment.

## What gets deployed

- The web app: `web/` (Starlette app served by uvicorn) and `app/components.py`.
- The dashboard data `app/data/` (about 190 MB). It is **not in git** (project rule: never commit data), so the
  image must be built on a machine that has it: run `python -m app.build_data` first.
- The committed result tables in `reports/test/` and `reports/late_blight/`.
- A persistent volume mounted at `/data` for the user database (`/data/auth/users.db`).

## Security model (what the host must provide)

- **HTTPS is required.** The container speaks plain HTTP on port 8765 and must only be reachable through an
  HTTPS proxy: the hosting platform's own TLS, or a reverse proxy such as Caddy.
- The image sets `HL_SECURE_COOKIES=1`: the session cookie is `__Host-hl_session` with `Secure`, `HttpOnly`,
  `SameSite=Strict`, and responses carry `Strict-Transport-Security`. Over plain HTTP the browser will not send
  the cookie, so sign-in will appear to fail. This is deliberate.
- uvicorn runs with `--proxy-headers --forwarded-allow-ips *` so the app sees the real scheme (https) and client
  IP from the proxy. That trusts any proxy, so **never expose port 8765 directly to the internet**.
- Accounts are invite-only, created by an admin on the server (below). There is no sign-up page.

## Steps

1. On a machine with the data:
   ```
   python -m app.build_data
   docker build -t hyperlocal-dashboard .
   ```
2. Run it with a named volume for the user database (example for a plain server):
   ```
   docker run -d --name hyperlocal -p 127.0.0.1:8765:8765 -v hyperlocal-auth:/data --restart unless-stopped hyperlocal-dashboard
   ```
   Binding to `127.0.0.1` keeps the container reachable only from the proxy on the same machine.
3. HTTPS with Caddy (automatic certificates), `/etc/caddy/Caddyfile`:
   ```
   forecasts.example.org {
       reverse_proxy 127.0.0.1:8765
   }
   ```
   On a hosting platform instead, point its HTTPS service at container port 8765 and attach a persistent
   volume at `/data`.
4. Create the first admin inside the container (the password is typed at a hidden prompt):
   ```
   docker exec -it hyperlocal python -m web.users add <your-name> --role admin
   docker exec -it hyperlocal python -m web.users add <supervisor> --role viewer
   ```
5. Check after deploying:
   - `https://<site>/healthz` returns `ok`; `https://<site>/` redirects to `/login`;
   - signing in works; in the browser's developer tools the cookie shows `Secure`, `HttpOnly`, `SameSite=Strict`;
   - the response headers include `Content-Security-Policy` and `Strict-Transport-Security`;
   - a viewer account gets "admins only" (403) at `/api/admin/users`;
   - `docker exec -it hyperlocal python -m web.users audit` lists the sign-ins.

## Operations

- Back up the volume (it holds `users.db`; losing it means re-creating accounts, nothing else).
- Reset a password: `python -m web.users reset-password <name>` (signs that user out everywhere).
- Remove access: `python -m web.users disable <name>` (immediate; their sessions end).
- New data (for example a new test period): rebuild `app/data`, rebuild and redeploy the image; the volume with
  the accounts is kept.
