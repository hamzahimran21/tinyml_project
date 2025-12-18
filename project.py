# ===========================================================================
# FMCW Radar Presence Detection Logger (FINAL, FIXED)
# - Range–Doppler processing
# - Static clutter removal
# - Presence energy feature per frame
# - 10 second logging duration
# - Graceful Ctrl+C handling
# ===========================================================================

import csv
import signal
import time
from datetime import datetime

import numpy as np
from ifxradarsdk import get_version
from ifxradarsdk.fmcw import DeviceFmcw
from ifxradarsdk.fmcw.types import (
    FmcwSimpleSequenceConfig,
    FmcwSequenceChirp,
)

# -------------------------
# Global control flag
# -------------------------
running = True


def handle_interrupt(signum, frame):
    global running
    print("\nInterrupt received. Stopping safely...")
    running = False


signal.signal(signal.SIGINT, handle_interrupt)

# -------------------------
# Radar configuration
# -------------------------
config = FmcwSimpleSequenceConfig(
    frame_repetition_time_s=307.325e-3,
    chirp_repetition_time_s=500e-6,
    num_chirps=64,  # Required for Doppler
    tdm_mimo=False,
    chirp=FmcwSequenceChirp(
        start_frequency_Hz=59e9,
        end_frequency_Hz=61e9,
        sample_rate_Hz=2e6,
        num_samples=128,
        rx_mask=1,
        tx_mask=1,
        tx_power_level=31,
        lp_cutoff_Hz=500_000,
        hp_cutoff_Hz=80_000,
        if_gain_dB=30,
    ),
)

# -------------------------
# CSV setup
# -------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_filename = f"radar_presence_{timestamp}.csv"

csv_file = open(csv_filename, mode="w", newline="")
csv_writer = csv.writer(csv_file)

csv_writer.writerow([
    "frame_idx",
    "rx_idx",
    "presence_energy",
])

# -------------------------
# Device acquisition
# -------------------------
with DeviceFmcw() as device:
    print("Radar SDK Version:", get_version())
    print("UUID of board:", device.get_board_uuid())
    print("Sensor:", device.get_sensor_type())

    sequence = device.create_simple_sequence(config)
    device.set_acquisition_sequence(sequence)

    frame_idx = 0
    start_time = time.time()
    LOG_DURATION_S = 20

    print("\nLogging PRESENCE data for 10 seconds...\n")

    while running and (time.time() - start_time) < LOG_DURATION_S:
        frame_contents = device.get_next_frame()

        for frame in frame_contents:
            # frame shape: [rx, chirps, samples]
            num_rx, num_chirps, num_samples = frame.shape

            for rx in range(num_rx):
                # -----------------------------
                # 1. Static clutter removal
                # -----------------------------
                frame_rx = frame[rx, :, :]
                frame_rx = frame_rx - np.mean(frame_rx, axis=0, keepdims=True)

                # -----------------------------
                # 2. Range FFT
                # -----------------------------
                range_fft = np.fft.fft(frame_rx, axis=1)
                range_fft = range_fft[:, :num_samples // 2]

                # -----------------------------
                # 3. Doppler FFT
                # -----------------------------
                doppler_fft = np.fft.fft(range_fft, axis=0)
                rd_map = np.abs(doppler_fft)

                # -----------------------------
                # 4. Presence feature
                # -----------------------------
                presence_energy = float(np.sum(rd_map))

                csv_writer.writerow([
                    frame_idx,
                    rx,
                    presence_energy,
                ])

        frame_idx += 1
        csv_file.flush()

# -------------------------
# Cleanup
# -------------------------
csv_file.flush()
csv_file.close()

print(f"\nData logging complete. File saved as: {csv_filename}")

