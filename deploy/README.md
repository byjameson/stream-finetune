# Deploying the spiking LM server

The app is a single Python process with one GPU lock. Put TLS and rate limits in
front of it; do not expose the Python `http.server` directly to the internet.

## 1. Install

On the host:

```bash
python3 -m venv /opt/spiking/venv
/opt/spiking/venv/bin/pip install streamft==0.6.4
```

Copy or train a checkpoint:

```bash
/opt/spiking/venv/bin/streamft spiking train \
  --data /opt/spiking/data/ts_train.pt \
  --max-tokens 0 \
  --valid /opt/spiking/data/ts_valid.pt \
  --out /opt/spiking/checkpoints/spiking_4m_full \
  --steps 1000 --batch 16 --seq 128 --dim 256 --layers 1 \
  --adapt 0.1 --tokenizer /opt/spiking/data/ts_bpe8k.json
```

The checkpoint directory must contain `model.pt` and `tokenizer.json`.

## 2. systemd

```bash
sudo useradd --system --home /opt/spiking --shell /usr/sbin/nologin spiking
sudo chown -R spiking:spiking /opt/spiking
sudo cp deploy/spiking.service /etc/systemd/system/
sudo install -m 600 /dev/null /etc/spiking.env
printf 'SPIKING_API_KEY=%s\n' "$(openssl rand -hex 24)" | sudo tee /etc/spiking.env
sudo systemctl daemon-reload
sudo systemctl enable --now spiking
```

The unit binds to `127.0.0.1:8011`; the reverse proxy is the only public entry.

## 3. TLS and rate limiting

Use either:

- `deploy/Caddyfile` for automatic HTTPS, or
- `deploy/nginx-spiking.conf` for nginx + certbot.

For nginx, adjust the `limit_req_zone` rate to your traffic. Caddy has no
built-in per-IP rate limiter in the default build; use nginx or a Caddy plugin
if you need it.

## 4. Smoke test

```bash
curl -s https://spiking.example.com/health
curl -s https://spiking.example.com/metrics

KEY=$(sudo awk -F= '/SPIKING_API_KEY/{print $2}' /etc/spiking.env)
curl -s -X POST https://spiking.example.com/generate \
  -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Once upon a time","max_new_tokens":32}'
```

`/health`, `/metrics` and `/v1/models` are public; the two POST endpoints require
the key. Keep the key out of browser code and logs.

## 5. GPU operations

- The server uses a process-wide lock, so requests are serialized. One process
  is enough for a 6 GB card.
- For multiple GPUs, run one process per GPU on different ports and put a load
  balancer in front. Do not run two processes on the same GPU.
- Cap `max_new_tokens` at the proxy or in the application; the app already caps
  it at 256.
- Watch `/metrics` (`requests`, `generated_tokens`) and add proxy latency/error
  metrics.
- Keep checkpoints immutable; put the version/step in the checkpoint name and
  roll forward by restarting with the new directory.

## 6. Honest quality note

The reference 4.3M model is a TinyStories model. It is suitable as a demo or a
research service, not as a general assistant. For a production-quality model,
train a 20M+ model on 100M+ domain tokens and evaluate it on a held-out set
before exposing it.
