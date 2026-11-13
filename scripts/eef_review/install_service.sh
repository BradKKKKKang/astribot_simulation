#!/usr/bin/env bash
# Install only this repository's EEF service into the current user's systemd.
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 - <<'PY'
from pathlib import Path
from src.eef_review.common import REPO, load_config
c = load_config()
cache = Path(c['cache'])
if not (cache / 'runtime.json').exists():
    raise SystemExit('Run scripts/eef_review/review start once before installing the persistent service.')
unit = f'''[Unit]
Description=Astribot EEF review gateway and SSH tunnel
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory={REPO}
ExecStart=/usr/bin/python3 -m src.eef_review.supervisor local --config {cache}/runtime.json --state {cache}/run
Restart=on-failure
RestartSec=3
TimeoutStopSec=20

[Install]
WantedBy=default.target
'''
(cache / 'astribot-eef-review.service').write_text(unit)
PY
systemctl --user stop astribot-eef-review.service
systemctl --user link "$PWD/outputs/eef_review/astribot-eef-review.service"
systemctl --user daemon-reload
systemctl --user enable --now astribot-eef-review.service
