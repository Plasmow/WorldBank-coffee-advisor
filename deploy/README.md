# Deploying on your own VM

Assumes Debian or Ubuntu, a public IP, and ports 80 and 443 reachable.
Nothing here needs the AI stack: with `USE_REAL_AI` unset the server answers
from the keyword chain in about 60 MB.

```bash
# 1. user and code
sudo adduser --system --group --home /opt/coffee-advisor coffee
sudo -u coffee git clone -b main https://github.com/Plasmow/WorldBankAgriculture.git /opt/coffee-advisor
cd /opt/coffee-advisor

# 2. python and dependencies (20 MB)
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo -u coffee uv venv --python 3.12
sudo -u coffee uv pip install -r requirements.txt

# 3. secrets, next to the code, never in git
sudo -u coffee tee /opt/coffee-advisor/.env > /dev/null <<'ENV'
AT_USERNAME=sandbox
AT_API_KEY=your-sandbox-key
AGENT_PHONE=+256700000099
FRONTEND_ORIGIN=https://your-page.lovable.app
DB_PATH=/opt/coffee-advisor/data/app.db
ENV
sudo chmod 600 /opt/coffee-advisor/.env

# 4. run it as a service
sudo cp deploy/coffee-advisor.service /etc/systemd/system/
sudo systemctl enable --now coffee-advisor
curl localhost:8000/health

# 5. HTTPS
sudo apt install -y caddy
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit the hostname first
sudo systemctl reload caddy
curl https://<your-host>/health
```

Then point Africa's Talking at `https://<your-host>/sms` (and `/ussd`), and
open the Lovable page with `?api=https://<your-host>`.

## Day to day

```bash
sudo systemctl restart coffee-advisor    # after a git pull or an .env change
sudo journalctl -u coffee-advisor -f     # live logs, including every SMS sent
```

`uvicorn --reload` is for a laptop. systemd restarts the process on its own,
so a crash at 3 a.m. does not end the demo.

## When the AI chain is ready

```bash
sudo -u coffee uv pip install -r requirements-ai.txt   # 125 MB
```
Add `USE_REAL_AI=1` to `.env` (the translator is fetched from the Hub on
first use, or set `TRANSLATOR_REPO`), then
restart. Budget 1.5 GB of RAM for the translator, and check `/health` says
`"loaded": true` before believing it.
