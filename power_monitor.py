#!/usr/bin/env python3
"""Minimal blue-team power monitor: detect ML training by sustained GPU power draw."""

import time
import sqlite3
from collections import deque

INA3221_PATH = "/sys/bus/i2c/drivers/ina3221/1-0040/iio:device0"
DB_PATH = "/opt/jetson/monitor.db"

# Tuning parameters
POLL_INTERVAL = 1.0          # seconds
WINDOW_SIZE = 30             # samples (30s window at 1Hz)
POWER_THRESHOLD_MW = 5000    # GPU power above this suggests active compute
SUSTAINED_FRACTION = 0.8     # fraction of window that must exceed threshold

def read_gpu_power_mw():
    """Read GPU power in milliwatts from INA3221 channel 0."""
    try:
        with open(f"{INA3221_PATH}/in_power0_input") as f:
            return float(f.read().strip())
    except (IOError, ValueError):
        return None

def main():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS power_log (
        timestamp REAL, gpu_mw REAL, training_detected INTEGER
    )""")

    window = deque(maxlen=WINDOW_SIZE)
    training_active = False

    while True:
        power = read_gpu_power_mw()
        if power is None:
            time.sleep(POLL_INTERVAL)
            continue

        window.append(power)

        # Check if enough of the recent window is above threshold
        if len(window) == WINDOW_SIZE:
            high_count = sum(1 for p in window if p > POWER_THRESHOLD_MW)
            fraction = high_count / WINDOW_SIZE
            now_training = fraction >= SUSTAINED_FRACTION

            if now_training and not training_active:
                print(f"[ALERT] Training detected — "
                      f"{fraction:.0%} of last {WINDOW_SIZE}s above {POWER_THRESHOLD_MW} mW")
            elif not now_training and training_active:
                print(f"[INFO] Training appears to have stopped")

            training_active = now_training
        else:
            training_active = False

        conn.execute("INSERT INTO power_log VALUES (?,?,?)",
                     (time.time(), power, int(training_active)))
        conn.commit()
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
