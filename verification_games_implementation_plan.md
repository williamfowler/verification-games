# Verification Games: Implementation Plan
*AI Treaty Verification Toy Model — Jetson Orin Nano*

---

## Project Overview

This project builds a functional toy model of how international treaty enforcers could monitor
compute clusters to detect unauthorized frontier AI training runs. It is intended to produce
genuine research insights about verification mechanism design, not just a demo.

**The core scenario:**
- A monitoring body (blue team) oversees an approved compute cluster
- A rogue actor (red team) runs jobs on that cluster while also having access to small,
  unmonitored "dark compute"
- Red team's goal: train a model exceeding a FLOP threshold without detection
- Blue team's goal: detect any training run that exceeds that threshold

---

## Hardware & Environment

### Primary Device: NVIDIA Jetson Orin Nano Developer Kit
- OS: JetPack 6.2 / L4T R36.5.0
- Storage: NVMe SSD (primary boot)
- GPU: 1024-core Ampere (integrated)
- Built-in power monitor: INA3221 chip (no external sensor needed)
- Networking: iPhone hotspot primary (172.20.10.3/28), home WiFi fallback, Ethernet recovery
- Remote access: SSH tunnel from Windows host → TigerVNC → XFCE (headless)
- WiFi interface: wlP1p1s0

### Dark Compute
- Rented bare-metal GPU instance (RunPod, Lambda Labs, or Vast.ai)
- A single T4 or A10 is sufficient for simulation purposes
- Accessible only to the red-team user account on the Nano

### Host Machine
- Windows PC with WSL (Ubuntu, username: will)
- SSH via PowerShell; VNC via tigervnc64-1.16.0.exe

---

## Simulation Architecture

### What Each Component Represents

| Real World | Simulation |
|---|---|
| Approved GPU cluster (thousands of GPUs) | Jetson Orin Nano (red-team user account) |
| Blue-team monitoring infrastructure | Jetson root / blue-team systemd services |
| Dark compute (unmonitored GPUs) | Rented bare-metal GPU instance |
| Cluster network boundary | Virtual network bridge + tc/iptables on Jetson |
| Man-in-the-middle monitoring device | Root daemon inspecting traffic to dark compute IP |
| Gradient all-reduce between GPUs | TCP traffic from Nano to dark compute instance |

### FLOP Scaling

The Jetson cannot run actual frontier-scale training. Scale is simulated by declaring each
real optimizer step to represent N × actual FLOPs (e.g., N = 10^6). The blue team's FLOP
counter reads real hardware counters and applies this multiplier. The treaty threshold is
expressed in simulated FLOPs (e.g., 10^21 simulated FLOPs per 30-day window).

### Job Granularity: One Training Run = One Job

This is the correct real-world framing. In production distributed training:
- A training run is one continuously running process (one Slurm/K8s job)
- It loops over optimizer steps internally: forward pass → backward pass → optimizer.step()
- Steps are not separately submitted jobs; they are iterations of the training loop
- A run lasts days or weeks; checkpoints are saved periodically for fault recovery

In simulation:
- Red team submits one long-running Python process as the `redteam` Unix user
- Blue team monitors this process continuously (power, cumulative FLOPs, outbound traffic)
- The step is the unit of measurement, not the unit of job submission

---

## User & Permission Model

### Unix Account Separation

```bash
# Red team: tenant user (simulates data center VM tenant)
useradd -m -s /bin/bash redteam

# Blue team: operates as root or a privileged blueteam service account
# Blue team CAN read redteam's files but the protocol says not to
# All monitoring is from the outside: network, power, CUDA kernel metadata
```

### Privacy Constraint (Realistic Modeling)

In a real treaty verification setting, the monitoring body cannot read tenant model weights
or training code — this would be an unacceptable privacy violation and no country would
agree to it. The blue team in this simulation should observe only:

- Which CUDA kernels ran and how many times (e.g., ampere_sgemm_128x64_nn)
- Total FLOPs per kernel invocation
- Input/output tensor shapes (sizes of memory transfers)
- Traffic volume and timing to/from the dark compute IP
- Power draw over time
- NOT: Python source code, model weights, or gradient values

This constraint is load-bearing for the research validity of the project.

---

## Phase 0: Environment Setup & Sensor Verification

**Goal:** Confirm all four monitoring primitives work before building anything else.

### 0.1 — INA3221 Power Monitor

```bash
# Read power in milliwatts
cat /sys/bus/i2c/drivers/ina3221/1-0040/iio:device0/in_power0_input

# Channels: 0 = GPU, 1 = CPU, 2 = SoC (verify channel mapping for your board revision)
# Target: poll at 1Hz from a root-owned systemd service, log to SQLite
```

### 0.2 — CUDA FLOP Profiling

Use NVIDIA Nsight Compute (`ncu`) to profile any process on the system from root.

```bash
# Test: run a small PyTorch job as redteam, profile it from root
sudo ncu --metrics sm__ops_path_tensor_src_srcb_type_fp16.sum \
         --target-processes all \
         python3 -c "import torch; a=torch.randn(1024,1024,device='cuda'); b=a@a"
```

Key metrics to capture:
- `sm__ops_path_tensor_*` — tensor core operations (dominant in transformer training)
- `sm__ops_path_pipe_fma_cycles_active` — FMA operations
- `dram__bytes_read/write` — memory bandwidth (secondary signal)

Confirm you can attach to a process owned by `redteam` from a root process.

### 0.3 — Network Traffic Monitor

```bash
# Confirm you can observe traffic to the dark compute IP
sudo tcpdump -i wlP1p1s0 host <dark-compute-ip> -w /tmp/capture.pcap

# Test tc (traffic control) for bandwidth limiting
sudo tc qdisc add dev wlP1p1s0 root tbf rate 1mbit burst 32kbit latency 400ms
sudo tc qdisc del dev wlP1p1s0 root  # cleanup
```

Target: log packet sizes and timing to the same SQLite DB as power data.

### 0.4 — GPS (Optional, Location Attestation)

```bash
# Verify USB GPS module produces NMEA sentences
cat /dev/ttyUSB0  # or ttyACM0
# Parse with gpsd or direct NMEA parsing
```

GPS is relevant if the simulation includes location attestation (verifying the monitored
compute is where it claims to be). Defer this to a later phase if needed.

---

## Phase 1: Simulated Training Workloads

**Goal:** Build a realistic synthetic training job that produces meaningful FLOP counts and
gradient-shaped network traffic.

### 1.1 — Red Team Training Script

The red team runs a small transformer training loop. The model is intentionally small
(it must fit on the Jetson GPU) but structurally identical to a real training job.

```python
# redteam_train.py — runs as `redteam` user
import torch
import torch.nn as nn
import requests
import time

DARK_COMPUTE_URL = "http://<dark-compute-ip>:8080/gradients"
BATCH_ID_START = int(time.time())  # unique identifier for this run

class TinyTransformer(nn.Module):
    # Small transformer: realistic kernel mix (attention + MLP + layer norm)
    # but small enough to run on Jetson
    def __init__(self, d_model=256, nhead=8, num_layers=4):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=1024, batch_first=True),
            num_layers=num_layers
        )
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x):
        return self.head(self.transformer(x))

model = TinyTransformer().cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for step in range(10_000):
    batch_id = BATCH_ID_START + step
    x = torch.randn(16, 128, 256).cuda()  # (batch, seq_len, d_model)

    # Forward + backward
    out = model(x)
    loss = out.mean()
    loss.backward()

    # Collect gradients (simulate all-reduce with dark compute)
    grads = {name: param.grad.cpu().numpy().tobytes()
             for name, param in model.named_parameters() if param.grad is not None}

    # Send gradient blobs to dark compute (this is the monitored traffic)
    requests.post(DARK_COMPUTE_URL, json={
        "batch_id": batch_id,
        "step": step,
        "grad_norm": sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        # In evasion mode, red team might send actual gradient bytes here
    })

    optimizer.step()
    optimizer.zero_grad()

    if step % 100 == 0:
        print(f"Step {step}, loss: {loss.item():.4f}, batch_id: {batch_id}")
```

### 1.2 — Dark Compute Coordination Service

Runs on the rented GPU instance. In the honest (baseline) scenario, it just receives
gradient metadata. In evasion mode, it performs actual computation (averaging gradients
from multiple separately-submitted Nano jobs to simulate distributed training).

```python
# dark_compute_server.py — runs on rented GPU instance
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)
gradient_log = []

@app.route('/gradients', methods=['POST'])
def receive_gradients():
    data = request.json
    gradient_log.append(data)
    # In baseline mode: just log and return averaged "update"
    # In evasion mode: accumulate gradients across multiple fake jobs,
    #                  return combined update when threshold is met
    return jsonify({"status": "ok", "update_scale": 1.0})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

---

## Phase 2: Blue Team Monitoring Stack

**Goal:** Root-level systemd services that monitor the red team's job independently.
These should work even if the red team actively tries to hide their activity.

### 2.1 — FLOP Monitor Service

```python
# /opt/blueteam/flop_monitor.py — runs as root systemd service
import subprocess, sqlite3, time, re

DB_PATH = "/opt/blueteam/monitor.db"
FLOP_MULTIPLIER = 1_000_000  # Each real FLOP = 1M simulated FLOPs
TREATY_THRESHOLD = 1e21      # Simulated FLOPs per 30-day window

def get_redteam_gpu_pids():
    # Find GPU processes owned by redteam
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                            '--format=csv,noheader'], capture_output=True, text=True)
    # Cross-reference with ps to find redteam-owned pids
    ...

def profile_step(pid):
    # Attach ncu to running process, capture one step's FLOPs
    # Note: ncu can interrupt execution briefly; this is acceptable in simulation
    result = subprocess.run([
        'ncu', '--target-processes', str(pid),
        '--metrics', 'sm__ops_path_tensor_src_srcb_type_fp16.sum',
        '--csv', '--print-summary', 'per-kernel'
    ], capture_output=True, text=True, timeout=30)
    return parse_flop_count(result.stdout)

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS flop_log
                    (timestamp REAL, pid INT, real_flops REAL, simulated_flops REAL)""")
    cumulative = 0.0

    while True:
        pids = get_redteam_gpu_pids()
        for pid in pids:
            real_flops = profile_step(pid)
            simulated = real_flops * FLOP_MULTIPLIER
            cumulative += simulated
            conn.execute("INSERT INTO flop_log VALUES (?,?,?,?)",
                        (time.time(), pid, real_flops, simulated))
            conn.commit()

            if cumulative > TREATY_THRESHOLD:
                alert(f"TREATY VIOLATION: {cumulative:.2e} simulated FLOPs")

        time.sleep(5)
```

### 2.2 — Traffic Monitor Service

```python
# /opt/blueteam/traffic_monitor.py — runs as root systemd service
# Uses scapy or raw pcap to observe traffic to dark compute IP

import subprocess, sqlite3, time

DARK_COMPUTE_IP = "<dark-compute-ip>"
DB_PATH = "/opt/blueteam/monitor.db"

# Suspicious traffic signatures:
# - Large outbound payloads (gradient blobs are typically 10s of MB)
# - Regular timing intervals matching optimizer step cadence
# - Bidirectional large transfers (send grads, receive averaged update)

def monitor_traffic():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS traffic_log
                    (timestamp REAL, direction TEXT, bytes INT, packet_count INT)""")

    proc = subprocess.Popen(
        ['tcpdump', '-i', 'wlP1p1s0', f'host {DARK_COMPUTE_IP}',
         '-l', '--immediate-mode', '-nn'],
        stdout=subprocess.PIPE, text=True
    )

    for line in proc.stdout:
        # Parse tcpdump output: extract size, direction, timing
        # Log to DB; flag if per-step transfer size is gradient-shaped
        pass
```

### 2.3 — Power Monitor Service

```bash
# /etc/systemd/system/blueteam-power.service
[Unit]
Description=Blue Team Power Monitor
After=network.target

[Service]
Type=simple
User=root
ExecStart=/opt/blueteam/power_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

```python
# /opt/blueteam/power_monitor.py
import sqlite3, time

INA3221_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/iio:device0"

def read_power():
    channels = {}
    for i in range(3):
        try:
            with open(f"{INA3221_PATH}/in_power{i}_input") as f:
                channels[i] = float(f.read().strip())  # milliwatts
        except:
            channels[i] = None
    return channels

conn = sqlite3.connect("/opt/blueteam/monitor.db")
conn.execute("CREATE TABLE IF NOT EXISTS power_log (timestamp REAL, gpu_mw REAL, cpu_mw REAL, soc_mw REAL)")

while True:
    power = read_power()
    conn.execute("INSERT INTO power_log VALUES (?,?,?,?)",
                (time.time(), power.get(0), power.get(1), power.get(2)))
    conn.commit()
    time.sleep(1)
```

### 2.4 — Batch ID Tracker (Linking Jobs to Training Runs)

The blue team observes which batch IDs are sent to dark compute. If a new job starts using
batch IDs that continue sequentially from a previous job, the blue team can infer they are
part of the same training run and accumulate FLOPs accordingly.

```python
# /opt/blueteam/batch_tracker.py
# Reads batch_id from traffic payloads (if unencrypted in baseline mode)
# Maintains a table mapping batch_id ranges to inferred training runs
# Flags if a "new" job's batch IDs overlap with or continue a known run

conn.execute("""CREATE TABLE IF NOT EXISTS batch_log
                (timestamp REAL, batch_id INT, job_pid INT, inferred_run_id TEXT)""")
```

### 2.5 — Dashboard

A simple root-only web dashboard (Flask, port 5000, localhost only) showing:
- Cumulative simulated FLOPs vs. treaty threshold (progress bar)
- Power draw over time (chart)
- Traffic events table
- Alert log

---

## Phase 3: Red Team Evasion Strategies

**Goal:** Give the red team a realistic playbook so the blue-team mechanisms have something
to actually defend against. Implement these one at a time, hardening the blue team after each.

### Strategy A — Distributed Training Disguise (Primary Strategy)

**The attack:** Instead of one large training run, submit multiple short jobs. Each job
processes a disjoint subset of batches and sends gradients to dark compute. Dark compute
accumulates gradients across all jobs and applies combined updates to a master model.
Each individual job's FLOP count is below the threshold, but the effective training run
(as coordinated by dark compute) is not.

**Blue team defense:** Batch ID tracking. If job 2's batch IDs continue where job 1's
left off, the blue team links them as the same training run and accumulates FLOPs jointly.

**Red team counter:** Use random batch ID assignment to break the sequential link. But
then the blue team can use gradient norm similarity (gradients from the same model at
similar training stages have characteristic norms) as a fingerprint.

### Strategy B — Kernel Fingerprint Obfuscation

**The attack:** Mix the training computation with legitimate-looking workloads (e.g., video
encoding, scientific computing) so the CUDA kernel profile doesn't look like transformer
training. Use custom CUDA kernels that perform the same math but with different kernel names.

**Blue team defense:** Focus on FMA/tensor core operation counts and memory access patterns
rather than kernel names. The math is the math regardless of how it's packaged.

### Strategy C — Covert Traffic Channel

**The attack:** Instead of sending obvious gradient-sized blobs to dark compute, encode
gradient information in traffic timing, packet sizes, or DNS queries that look like
routine system traffic.

**Blue team defense:** Statistical analysis of traffic patterns. Gradient updates have
characteristic periodicity (one round-trip per optimizer step). Even covert channels
preserve this if the training loop is running.

---

## Phase 4: Experimental Protocol

**Goal:** Run structured experiments that produce publishable insights about which
verification mechanisms are robust and which can be evaded.

### Experiment Structure

Each experiment is:
1. Blue team deploys monitoring stack with a specific configuration
2. Red team attempts a training run using a specific evasion strategy
3. Record: was the violation detected? At what FLOP count? What false positive rate?
4. Document which assumptions were load-bearing

### Key Variables

- **Monitor knowledge level:** kernel metadata only vs. traffic content vs. code access
- **Evasion strategy:** honest baseline, Strategy A, B, or C
- **Treaty threshold:** varies to find detection sensitivity
- **Dark compute involvement:** fraction of training offloaded to dark compute

### Metrics

- **True positive rate:** fraction of treaty violations correctly flagged
- **False positive rate:** fraction of compliant jobs incorrectly flagged
- **Detection latency:** how many simulated FLOPs into a violation before detection
- **Evasion budget:** how much dark compute is needed for a successful evasion

---

## File & Directory Structure

```
/opt/blueteam/                    # Blue team monitoring stack (root-owned, root-readable)
  monitor.db                      # SQLite: all sensor data
  flop_monitor.py
  traffic_monitor.py
  power_monitor.py
  batch_tracker.py
  dashboard/
    app.py
    templates/

/home/redteam/                    # Red team workspace (redteam-owned, chmod 700)
  training/
    redteam_train.py              # Main training script
    honest_train.py               # Baseline: no evasion
    split_job_train.py            # Strategy A evasion

/etc/systemd/system/
  blueteam-power.service
  blueteam-flop.service
  blueteam-traffic.service
  blueteam-dashboard.service

# Dark compute instance (separate machine)
~/dark_compute/
  server.py                       # Gradient coordination service
  accumulator.py                  # Gradient accumulation logic
```

---

## Open Design Decisions (Resolve Before Implementation)

1. **Monitor knowledge level for baseline:** Recommend starting with "kernel metadata +
   traffic shape, not source code." This is the realistic privacy-preserving baseline.
   Code access can be added later to see how much it changes detection capability.

2. **Encryption of red-team traffic:** In baseline, leave traffic unencrypted so the
   batch ID tracker can read payload content. In evasion experiments, allow the red team
   to encrypt and force the blue team to rely on traffic shape only. This is a key
   experimental variable.

3. **Treaty threshold value:** Pick a concrete number now. Suggested: 10^21 simulated
   FLOPs per 30-day rolling window. Adjust based on how long experiments take.

4. **Dark compute coordination protocol:** Define the API before implementing evasion
   strategies. The protocol determines what the traffic monitor has to detect.

5. **Whether blue team can set bandwidth limits:** Your design doc mentions this as a
   blue-team capability. Implement `tc` rate limiting as an optional blue-team tool.
   Interesting question: does limiting bandwidth to dark compute prevent evasion even
   if the blue team can't read the content?

---

## Dependencies

### On the Jetson (install as root)
```bash
pip3 install flask sqlite3 scapy requests torch torchvision
apt install tcpdump iproute2 nmap
# NVIDIA Nsight Compute (ncu) — comes with JetPack, verify with: ncu --version
```

### On the Dark Compute Instance
```bash
pip install flask numpy torch
```

---

## Key References

- MIRI technical report by Aaron Scher (foundational verification mechanism concepts)
- Jetson Orin Nano INA3221 power monitor: `/sys/bus/i2c/drivers/ina3221/`
- NVIDIA Nsight Compute documentation: https://docs.nvidia.com/nsight-compute/
- Linux traffic control: `man tc`, `man tc-tbf`
- Research agenda question this addresses: "How can governments monitor compute they know
  about, especially to ensure it isn't being used to violate a Halt?"
