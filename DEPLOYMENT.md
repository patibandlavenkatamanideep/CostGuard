# Additional Deployment Options

The main [README](README.md) covers Railway and self-host with Docker Compose. This page documents two additional options.

---

## Persistence — the writable data path

CostGuard writes one SQLite file at `COSTGUARD_DB_PATH` (default `/data/costguard_history.db`). That single file holds **all** persisted state:

- evaluation + proxy call history (the dashboard's charts),
- per-provider **circuit-breaker** state (so a tripped breaker survives a restart),
- **alert** cooldowns (so you aren't re-paged after a redeploy).

The process must be able to **write** to the directory containing `COSTGUARD_DB_PATH`. If the path is not on a persistent volume, all of the above resets on every restart/redeploy.

| Platform | Persistent? | How |
|----------|-------------|-----|
| **Docker Compose** | ✅ | The `costguard-data` named volume is mounted at `/data` on both `api` and `ui` (already configured in `docker-compose.yml`). |
| **Railway** | ✅ (with a volume) | Add a **Volume** in the service settings mounted at `/data`, then set `COSTGUARD_DB_PATH=/data/costguard_history.db`. Without the volume, `/data` is ephemeral. `railway.json` cannot declare volumes — this is a one-time dashboard step. |
| **Koyeb / HF Spaces (free)** | ❌ | No persistent disk on the free tiers — point `COSTGUARD_DB_PATH` at `/tmp/...`; history and runtime state reset on restart. Set `COSTGUARD_STATE_BACKEND=none` to skip persistence entirely and silence restore warnings. |

To verify persistence locally:

```bash
docker compose up -d
curl -s localhost:8000/health | jq .circuit_breakers   # trip/observe state
docker compose restart api
curl -s localhost:8000/health | jq .circuit_breakers   # state is restored from the volume
```

---

## Koyeb (free tier, no sleep, no CLI needed)

1. Go to [koyeb.com](https://www.koyeb.com/) → **Deploy** → **GitHub**
2. Select `patibandlavenkatamanideep/CostGuard`
3. Builder: **Dockerfile**, Port: **8501**, Health path: `/_stcore/health`
4. Add env vars: `SECRET_KEY`, `COSTGUARD_API_KEY` (`openssl rand -hex 32`), `PORT=8501`, `API_PORT=9000`, `COSTGUARD_DB_PATH=/tmp/costguard_history.db`, plus any provider keys
   - For a public, read-only demo with no key, set `COSTGUARD_ALLOW_UNAUTHENTICATED=true` instead of `COSTGUARD_API_KEY` (the app refuses to start in production with neither). See [SECURITY.md](SECURITY.md).
5. Click **Deploy**

**Free tier:** 512 MB RAM, 0.1 vCPU, always-on (no sleep). No persistent disk — SQLite resets on redeploy.

---

## Hugging Face Spaces (free, great ML community visibility)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Space SDK: **Docker**, Visibility: **Public**
3. In the Space settings → **Secrets**: add `SECRET_KEY`, `COSTGUARD_API_KEY` (or `COSTGUARD_ALLOW_UNAUTHENTICATED=true` for a public demo), `OPENAI_API_KEY`, etc.
4. Push the repo or link via GitHub integration

Add this header to your Space's `README.md`:

```yaml
---
title: CostGuard
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8501
pinned: false
---
```

**Free tier:** 2 vCPU, 16 GB RAM (CPU Basic Space). No persistent storage on free tier — SQLite resets on restart.
