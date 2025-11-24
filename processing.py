import numpy as np
from scipy.signal import find_peaks
from common import ADC_LSB, MAX_INPUT_RANGE, MICRO
from typing import Dict, List

def compute_settling_time(
    raw: np.ndarray,
    fs: float,
    tol_u_v: float,
    final_offset_us: float = 6.0,
    final_duration_us: float = 3.0,
    min_stable_samples: int = 10,
):
    """
    Compute settling time with separation of group delay and pure settling.
    
    Returns:
        trigger_idx: Index representing trigger time (t=0, assumed at start of step detection)
        response_idx: Index where ADC first responds to the step (end of group delay)
        settling_idx: Index where signal has settled within tolerance
        ts_us: Time step in microseconds
    """
    ts_us = MICRO / fs

    # Detect the step edge - this represents the trigger point (t=0)
    trigger_idx = None
    for i in range(1, len(raw)):
        if raw[i] - raw[i-1] > tol_u_v:
            trigger_idx = i - 1
            break
    if trigger_idx is None:
        return None, None, None, None

    # The point where we detect the edge is actually when the ADC responds
    # So the response starts at trigger_idx, but we can better detect it
    # by finding where the signal starts changing significantly
    response_idx = trigger_idx

    # Estimate the final value from a later window
    i0 = int(trigger_idx + final_offset_us  / ts_us)
    i1 = int(i0          + final_duration_us/ ts_us)
    if i1 > len(raw):
        return None, None, None, None
    final_val = float(raw[i0:i1].mean())

    # Find the first index i where the next N samples are all within ±tol of final_val
    settling_idx = None
    last_start = len(raw) - min_stable_samples + 1
    for i in range(response_idx+1, last_start):
        block = raw[i : i + min_stable_samples]
        if np.all(np.abs(block - final_val) < tol_u_v):
            # Mark settling point at the end of the stable block
            settling_idx = i
            break

    if settling_idx is None:
        return None, None, None, None

    return trigger_idx, response_idx, settling_idx, ts_us

def compute_mean_settling_time(
    raw_runs: list[np.ndarray],
    trigger_idxs: list[int],
    response_idxs: list[int],
    settling_idxs: list[int],
    ts_us: float,
    pad: int = 2
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Compute mean settling metrics across multiple runs.
    
    Returns:
        mean_group_delay: Mean time from trigger to response (us)
        mean_settling: Mean time from response to settled (us)
        time_vec: Time vector for the mean trace
        mean_trace: Mean voltage trace across all runs
    """
    t_triggers = np.array(trigger_idxs) * ts_us
    t_responses = np.array(response_idxs) * ts_us
    t_settled = np.array(settling_idxs) * ts_us
    
    group_delays = t_responses - t_triggers
    settling_times = t_settled - t_responses
    total_times = t_settled - t_triggers

    mean_group_delay = group_delays.mean()
    mean_settling = settling_times.mean()
    mean_total = total_times.mean()

    # Find equal-length window across runs (aligned to trigger point)
    pre_samps = min(trigger_idxs)
    post_samps = min(len(raw) - end for raw, end in zip(raw_runs, settling_idxs))
    delta_samps = min(e - s for s, e in zip(trigger_idxs, settling_idxs))

    total_len = pre_samps + delta_samps + post_samps

    aligned = []
    for raw, trig_i in zip(raw_runs, trigger_idxs):
        # Extract segment around the trigger index (t=0)
        seg = raw[trig_i - pre_samps : trig_i - pre_samps + total_len]
        aligned.append(seg)
    aligned = np.vstack(aligned)

    mean_trace = aligned.mean(axis=0)

    # Time vector relative to trigger (t=0 at trigger)
    time_full = (np.arange(-pre_samps, -pre_samps + total_len) * ts_us)

    # Truncate to show from slightly before trigger to slightly after settled
    zero_idx = int(np.searchsorted(time_full, 0))
    end_idx = int(np.searchsorted(time_full, mean_total))

    start_i = max(0, zero_idx - pad)
    stop_i = min(len(time_full), end_idx + pad)

    time_trunc = time_full[start_i:stop_i]
    trace_trunc = mean_trace[start_i:stop_i]

    return mean_group_delay, mean_settling, time_trunc, trace_trunc

def find_spur_rms(freqs, spectrum, target_freq, span_hz=5e3):
    # Limit to a window around the target
    mask = np.abs(freqs - target_freq) <= span_hz
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        raise ValueError(f"No bins within ±{span_hz} Hz of {target_freq} Hz")

    # Find peaks in that window
    sub = spectrum[idxs]
    peaks, _ = find_peaks(sub, height=np.max(sub)*0.1)  # only peaks >10% of local max
    peaks = np.asarray(peaks)
    if peaks.size == 0:
        peak_idx = idxs[np.argmax(sub)]
    else:
        peak_idx = idxs[peaks[np.argmax(sub[peaks])]]

    v_peak  = spectrum[peak_idx]
    v_rms   = v_peak / np.sqrt(2)
    return v_rms, freqs[peak_idx]

def compute_dynamics(freq_axis, mag_vec, k_fund, passband_hz,
                     h=5, dc_bins=10):
    main_bins  = 15 # +/- main_bins-lobe bins

    mask = np.ones_like(mag_vec, dtype=bool)
    mask[:dc_bins] = False # DC guard
    mask[max(0, k_fund - main_bins):k_fund + main_bins + 1] = False  # Fundamental
    for h in range(2, h + 1): # H2...H5
        bin_h = h * k_fund
        if bin_h < len(mask):
            mask[max(0, bin_h - main_bins):bin_h + main_bins + 1] = False
    mask &= freq_axis <= passband_hz # Pass-band

    p1   = (mag_vec[k_fund - main_bins:k_fund + main_bins + 1] ** 2).sum()
    fund = np.sqrt(p1)
    spur = mag_vec[mask].max()
    sfdr = 20.0 * np.log10(fund / spur)

    ph = sum((mag_vec[h * k_fund - main_bins:
                        h * k_fund + main_bins + 1] ** 2).sum()
             for h in range(2, h + 1) if h * k_fund < len(mag_vec))
    thd   = 10.0 * np.log10(ph / p1) if ph > 0.0 else -np.inf

    pnd   = (mag_vec[mask] ** 2).sum()
    sinad = 10.0 * np.log10(p1 / pnd)
    enob  = (sinad - 1.76) / 6.02
    return sfdr, thd, sinad, enob

def dc_summary(tag: str,
               run_stats: Dict[str, List[dict]]) -> dict[str, float] | None:
    if not run_stats.get(tag):
        return None

    gain  = np.mean([r["gain"]        for r in run_stats[tag]])
    offset_uv = np.mean([r["offset_uV"]   for r in run_stats[tag]])
    inl_max_ppm = np.mean([r["max_inl_ppm"] for r in run_stats[tag]])
    inl_rms_lsb = np.mean([r["rms_inl_lsb"] for r in run_stats[tag]])
    inl_typ_ppm = np.mean([r["typ_inl_ppm"] for r in run_stats[tag]])

    return {
        "runs"          : len(run_stats[tag]),
        "gain_err_pct"  : (gain - 1.0) * 100.0,
        "offset_uV"     : offset_uv,
        "max_inl_ppm"   : inl_max_ppm,
        "typ_inl_ppm"   : inl_typ_ppm,
        "rms_inl_lsb"   : inl_rms_lsb,
    }

def analyze_linearity(x: np.ndarray,
                      y: np.ndarray,
                      label: str) -> dict[str, float | np.ndarray]:

    gain, offset = np.polyfit(x, y, 1)
    fit_line = gain * x + offset
    residual    = y - fit_line

    inl_lsb = residual / ADC_LSB
    inl_ppm = residual / (MAX_INPUT_RANGE * 2) * MICRO

    return {
        "label"      : label,
        "gain"       : gain,
        "offset_uV"  : offset * MICRO,
        "max_inl_ppm": np.max(np.abs(inl_ppm)),
        "typ_inl_ppm": np.mean(np.abs(inl_ppm)),
        "rms_inl_lsb": np.sqrt(np.mean(inl_lsb ** 2)),
        "fit_line"   : fit_line,
        "inl_lsb"    : inl_lsb,
    }
