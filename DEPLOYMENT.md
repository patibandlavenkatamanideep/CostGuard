# Additional Deployment Options

The main [README](README.md) covers Render, Fly.io, and self-host with Docker Compose. This page documents two additional free-tier options.

---

## Koyeb (free tier, no sleep, no CLI needed)

1. Go to [koyeb.com](https://www.koyeb.com/) → **Deploy** → **GitHub**
2. Select `patibandlavenkatamanideep/CostGuard`
3. Builder: **Dockerfile**, Port: **8501**, Health path: `/_stcore/health`
4. Add env vars: `SECRET_KEY`, `PORT=8501`, `API_PORT=9000`, `COSTGUARD_DB_PATH=/tmp/costguard_history.db`, plus any provider keys
5. Click **Deploy**

**Free tier:** 512 MB RAM, 0.1 vCPU, always-on (no sleep). No persistent disk — SQLite resets on redeploy.

---

## Hugging Face Spaces (free, great ML community visibility)

1. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
2. Space SDK: **Docker**, Visibility: **Public**
3. In the Space settings → **Secrets**: add `SECRET_KEY`, `OPENAI_API_KEY`, etc.
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
