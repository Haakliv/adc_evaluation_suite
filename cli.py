import argparse
import datetime
import logging
import os
import numpy as np
import time
from scipy.signal import welch
import math
from scipy.fft import rfft, rfftfreq
from multimeter import Dmm6500Controller
from common import (
    SLAVE_ODR_MAP,
    ADC_LSB,
    DEFAULT_STEP_VPP,
    KILO,
    MICRO,
)
from ace_client import (
    ACEClient,
    MAX_INPUT_RANGE,
    SINC_FILTER_MAP,
)
from generator import WaveformGenerator
from source import B2912A
from acquisition import capture_samples
from processing import (
    compute_settling_time,
    compute_mean_settling_time,
    find_spur_rms,
    compute_dynamics,
    dc_summary,
    analyze_linearity,
)
from plotting import (
    plot_agg_fft,
    plot_agg_histogram,
    plot_settling_time,
    plot_freq_response,
    plot_dc_linearity_summary,
    plot_fft_with_metrics,
)

ACE_HOST_DEFAULT = 'localhost:2357'
SDG_HOST_DEFAULT = '172.16.1.56'
B29_HOST_DEFAULT = '169.254.5.2'
DMM_IP_DEFAULT = '169.254.15.212'

LOW_PCT = 50.0

ADC_ODR_CODE_DEFAULT = 1  # 1.25 MHz
ADC_FILTER_CODE_DEFAULT = 2  # Sinc6

SAMPLES_DEFAULT = 131072
HIST_BINS_DEFAULT = 1000
SETTLING_THRESH_FRAC = 0.01
SWEEP_POINTS_DEFAULT = 50

SWITCHING_FREQ = 295400  # Hz, power-supply switching frequency for spur detection
FILTER_BW_FACTORS = 232630  # Hz  # Sinc6 BW factor (−3 dB) at 1.25MHz ODR for spur integration

DEFAULT_INL_STEPS = 4096  # 2 mV step over 8.192 V span
DEFAULT_RUNS = 3


# =============================================================================
# Measurement routines
# =============================================================================
# -- Noise floor --------------------------------------------------------------
def run_noise_floor(args, logger, ace):
    f = 0
    ace.setup_capture(args.samples, args.odr_code)
    odr = SLAVE_ODR_MAP[args.odr_code]
    filt = SINC_FILTER_MAP[args.filter_code]
    logger.info("Noise-floor: runs=%d, ODR=%.0f Hz, filter=%s",
                args.runs, odr, filt)

    welch_sum, stds, raw_runs = None, [], []
    for i in range(1, args.runs + 1):
        raw = capture_samples(
            ace, args.samples, ADC_LSB, output_dir=os.getcwd()
        )
        raw_runs.append(raw)
        mean = np.mean(raw)
        std = np.std(raw, ddof=0)
        ptp = np.ptp(raw)
        stds.append(std)

        # Welch PSD estimate
        f, psd = welch(raw, fs=odr,
                       nperseg=len(raw) // 4, noverlap=len(raw) // 8)
        # Ignore DC and Nyquist
        welch_sum = psd if welch_sum is None else welch_sum + psd
        # Calculate median noise spectral density (NSD)
        nsd_med = np.median(np.sqrt(psd)[1:-1])
        logger.info("Run %d: mean=%.3e V, std=%.3e V, ptp=%.3e V, "
                    "NSD_med=%.3e V/sqrt(Hz)", i, mean, std, ptp, nsd_med)

    rms = np.mean(stds)
    spread = np.std(stds, ddof=1)
    logger.info("RMS noise: %.3e V +- %.3e V", rms, spread)

    # Compute average PSD
    psd_avg = welch_sum / args.runs
    # Calculate median noise spectral density (NSD)
    nsd_avg = np.sqrt(psd_avg)
    # Calculate average noise spectral density (NSD)
    mag_avg = np.sqrt(psd_avg * odr / 2.0)
    # Calculate median NSD excluding DC and Nyquist
    med_nsd = np.median(nsd_avg[1:-1])
    logger.info("Median NSD: %.3e V/sqrt(Hz)", med_nsd)

    try:
        spur_rms_v, spur_freq = find_spur_rms(f, mag_avg, SWITCHING_FREQ, span_hz=10e3)
        spur_uvrms = spur_rms_v * MICRO
        logger.info(
            "Power-supply spur RMS @ %.1f kHz: %.3f uVrms",
            spur_freq / KILO, spur_uvrms
        )
    except ValueError as e:
        logger.warning("Spur detection failed: %s", e)

    if args.plot or args.show:
        if args.histogram:
            plot_agg_histogram(
                np.concatenate(raw_runs), args.hist_bins,
                args.runs, odr, filt,
                out_file='agg_hist.png', show=args.show
            )
        if args.fft:
            mag = np.sqrt(psd_avg * odr / 2)
            plot_agg_fft(
                f, mag, args.runs, odr, filt,
                out_file='agg_fft.png', show=args.show
            )


# -------------------------------------------------------------------------
#  Dynamic-performance test with external sine wave, SFDR, THD, SINAD, ENOB
# -------------------------------------------------------------------------
def run_sfdr(args, logger, ace):
    fs = SLAVE_ODR_MAP[args.odr_code]
    samples = args.samples
    passband_bw_hz = 69793  # -3 dB bandwidth in Hz at ODR_code 7 (375 kSPS)

    k_bin = int(round(args.freq * samples / fs))
    f_coh = k_bin * fs / samples  # Actual coherent frequency

    window_coefficients = [0.2712203606, 0.4334446123, 0.21800412,
         0.0657853433, 0.0107618673, 0.0007700125,
         0.0000136809]  # Coefficients for 7-term window
    # Ensure the coefficients sum to 1
    sample_indices = np.arange(samples)
    # Apply the window to the FFT
    window = np.sum([window_coefficients[m] * np.cos(2.0 * np.pi * m * (sample_indices - samples / 2) / samples)
                  for m in range(7)], axis=0)
    # Normalize the window
    window_correction_factor = window.sum() / samples

    logger.info("Dynamic performance test: tone %.0f Hz (coherent %.6f Hz), %.2f Vpp-diff, "
                "%d runs, ODR %.0f Hz, %d samples",
                args.freq, f_coh, args.amplitude,
                args.runs, fs, samples)

    if args.no_board:
        logger.warning("--no-board specified; skipping ADC capture")
        return None, None, None, None

    ace.setup_capture(samples, args.odr_code)

    spectra = []
    sfdr_runs = []
    thd_runs = []
    sinad_runs = []
    enob_runs = []

    freqs = rfftfreq(samples, 1.0 / fs)
    for _ in range(1, args.runs + 1):
        raw = capture_samples(ace_client=ace,
                              sample_count=samples,
                              output_dir=os.getcwd())
        # Apply the window to the raw data
        spec = rfft(raw * window)
        # Scale the FFT output
        fft_magnitude = 2.0 * np.abs(spec) / (samples * window_correction_factor) / np.sqrt(2.0)  # Vrms / bin
        # Find the coherent frequency bin
        spectra.append(fft_magnitude)

        # Calculate SFDR, THD, SINAD, ENOB for each run
        sfdr_i, thd_i, sinad_i, enob_i = compute_dynamics(freqs, fft_magnitude, k_bin, passband_bw_hz)
        sfdr_runs.append(sfdr_i)
        thd_runs.append(thd_i)
        sinad_runs.append(sinad_i)
        enob_runs.append(enob_i)

    # Compute the average spectrum across all runs
    fft_magnitude_avg = np.mean(spectra, axis=0)
    # Apply the window correction factor
    fft_magnitude_avg = np.maximum(fft_magnitude_avg, 1e-20)  # avoid log(0)

    # Calculate the frequency axis for the FFT
    # (freqs already defined above)
    sfdr, thd, sinad, enob = compute_dynamics(freqs, fft_magnitude_avg, k_bin, passband_bw_hz)

    # Calculate mean, std, and 95% CI for each metric
    def _stats(arr):
        arr = np.array(arr)
        mean = arr.mean()
        std = arr.std(ddof=1) if arr.size > 1 else 0.0
        ci95 = 1.96 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return mean, std, ci95

    sfdr_mean, sfdr_std, sfdr_ci = _stats(sfdr_runs)
    thd_mean, thd_std, thd_ci = _stats(thd_runs)
    sinad_mean, sinad_std, sinad_ci = _stats(sinad_runs)
    enob_mean, enob_std, enob_ci = _stats(enob_runs)

    logger.info("SFDR  : %.2f dB +- %.2f dB (95%% CI), std=%.2f dB over %d runs",
                sfdr_mean, sfdr_ci, sfdr_std, len(sfdr_runs))
    logger.info("THD   : %.2f dB +- %.2f dB (95%% CI), std=%.2f dB over %d runs",
                thd_mean, thd_ci, thd_std, len(thd_runs))
    logger.info("SINAD : %.2f dB +- %.2f dB (95%% CI), std=%.2f dB over %d runs",
                sinad_mean, sinad_ci, sinad_std, len(sinad_runs))
    logger.info("ENOB  : %.2f  +- %.2f  (95%% CI), std=%.2f over %d runs",
                enob_mean, enob_ci, enob_std, len(enob_runs))

    if args.plot or args.show:
        filt_name = SINC_FILTER_MAP[args.filter_code]
        plot_fft_with_metrics(
            freqs, fft_magnitude_avg, fs,
            sfdr, thd, sinad, enob,
            runs=args.runs,
            tone_freq=f_coh,
            amplitude_vpp=args.amplitude,
            filt=filt_name,
            out_file="sfdr_fft.png",
            show=args.show,
            xlim=(0.0, passband_bw_hz / KILO),
        )

    return sfdr, thd, sinad, enob



# -- Settling‑time ------------------------------------------------------------
def run_settling_time(args, logger, ace):
    odr = SLAVE_ODR_MAP[args.odr_code]
    filt = SINC_FILTER_MAP[args.filter_code]
    ts_us = MICRO / odr

    lsb_eff_u_v = 2 * MAX_INPUT_RANGE / 2 ** 17.3 # 17.3 ENOB, see thesis for why this is used

    logger.info("Settling test: ODR=%.0fHz, %s, runs=%d, Vpp=%.2f, freq=%.1fHz",
                odr, filt, args.runs, args.amplitude, args.frequency)

    ace.setup_capture(args.samples, args.odr_code)

    gen = WaveformGenerator(args.sdg_host, "PULSE", args.offset)
    gen.pulse_diff(args.frequency, args.amplitude,
                   low_percent=50.0, edge_time=2e-9, enable_trigger_out=True)
    time.sleep(0.5)

    raw_runs = []
    trigger_idxs = []
    response_idxs = []
    settling_idxs = []
    group_delay_times_ns = []
    settling_times_ns = []
    total_times_ns = []

    for run in range(1, args.runs + 1):
        raw = capture_samples(
            ace, args.samples, ADC_LSB,
            output_dir=os.getcwd()
        )
        raw_runs.append(raw)

        trig_idx, resp_idx, sett_idx, ts = compute_settling_time(
            raw, fs=odr, tol_u_v=lsb_eff_u_v
        )

        if trig_idx is None or resp_idx is None or sett_idx is None:
            logger.warning("Run %d: settling not detected", run)
            continue

        t_trigger = trig_idx * ts
        t_response = resp_idx * ts
        t_settled = sett_idx * ts
        
        group_delay_us = t_response - t_trigger
        settling_us = t_settled - t_response
        total_us = t_settled - t_trigger

        trigger_idxs.append(trig_idx)
        response_idxs.append(resp_idx)
        settling_idxs.append(sett_idx)
        group_delay_times_ns.append(group_delay_us * KILO)
        settling_times_ns.append(settling_us * KILO)
        total_times_ns.append(total_us * KILO)

        logger.info("Run %d: Group delay=%.2f us, Settling=%.2f us, Total=%.2f us",
                    run, group_delay_us, settling_us, total_us)

    gen.disable(1)
    gen.disable(2)

    if not trigger_idxs or not settling_idxs:
        logger.error("No valid measurements obtained")
        return

    # Compute and log the mean metrics across all runs
    mean_group_delay, mean_settling, time_vec, mean_seg = compute_mean_settling_time(
        raw_runs, trigger_idxs, response_idxs, settling_idxs, ts_us, pad=2
    )
    
    # Calculate statistics for group delay
    gd_arr = np.array(group_delay_times_ns)
    gd_mean_ns = gd_arr.mean()
    gd_std_ns = gd_arr.std(ddof=1) if gd_arr.size > 1 else 0.0
    gd_ci95_ns = 1.96 * gd_std_ns / np.sqrt(gd_arr.size) if gd_arr.size > 1 else 0.0
    
    # Calculate statistics for settling time
    st_arr = np.array(settling_times_ns)
    st_mean_ns = st_arr.mean()
    st_std_ns = st_arr.std(ddof=1) if st_arr.size > 1 else 0.0
    st_ci95_ns = 1.96 * st_std_ns / np.sqrt(st_arr.size) if st_arr.size > 1 else 0.0
    
    # Calculate statistics for total time
    tot_arr = np.array(total_times_ns)
    tot_mean_ns = tot_arr.mean()
    tot_std_ns = tot_arr.std(ddof=1) if tot_arr.size > 1 else 0.0
    tot_ci95_ns = 1.96 * tot_std_ns / np.sqrt(tot_arr.size) if tot_arr.size > 1 else 0.0

    # Log mean +-95% CI and std in us
    logger.info(
        "Mean group delay: %.2f us +- %.2f us (95%% CI), std=%.2f us over %d runs",
        gd_mean_ns * 1e-3, gd_ci95_ns * 1e-3, gd_std_ns * 1e-3, gd_arr.size
    )
    logger.info(
        "Mean settling time: %.2f us +- %.2f us (95%% CI), std=%.2f us over %d runs",
        st_mean_ns * 1e-3, st_ci95_ns * 1e-3, st_std_ns * 1e-3, st_arr.size
    )
    logger.info(
        "Mean total time: %.2f us +- %.2f us (95%% CI), std=%.2f us over %d runs",
        tot_mean_ns * 1e-3, tot_ci95_ns * 1e-3, tot_std_ns * 1e-3, tot_arr.size
    )

    if (args.plot or args.show) and raw_runs and trigger_idxs:
        plot_file = os.path.join(os.getcwd(), 'settling.png')
        plot_settling_time(
            raw_runs, trigger_idxs, response_idxs, settling_idxs,
            ts_us, time_vec, mean_seg, 
            mean_group_delay, mean_settling,
            filt, odr,
            args.frequency, args.amplitude, args.runs,
            out_file=plot_file,
            show=args.show
        )


def measure_tone(
        ace_client,
        wave_gen: "WaveformGenerator",
        freq_hz,
        odr_hz: float,
        amplitude: float,
        offset: float,
        settle_cycles: int,
        capture_cycles: int,
        logger,
        ch_pos: int = 1,
        ch_neg: int = 2,
) -> tuple[float, float]:
    wave_gen.sine_diff(freq_hz, amplitude, offset,
                       ch_pos=ch_pos, ch_neg=ch_neg)

    time.sleep(settle_cycles / freq_hz)

    samples = int(odr_hz * capture_cycles / freq_hz)
    logger.info(f"Capture @ {freq_hz:.1f} Hz = {samples} samples")

    if samples > SAMPLES_DEFAULT:
        # Shrink capture_cycles so it fits in one shot
        new_capture_cycles = int(SAMPLES_DEFAULT * freq_hz / odr_hz)
        logger.warning(
            "tone %.1f Hz: %d samples would exceed buffer; "
            "reducing capture_cycles from %d to %d",
            freq_hz, samples, capture_cycles, new_capture_cycles
        )
        samples = SAMPLES_DEFAULT

    raw = capture_samples(
        ace_client=ace_client,
        sample_count=samples,
        output_dir=os.getcwd()
    )

    vrms = float(np.sqrt(np.mean(raw.astype(np.float64) ** 2)))
    return freq_hz, vrms


# -- Frequency response ------------------------------------------------------
def run_freq_response(args, logger, ace):
    odr_hz = SLAVE_ODR_MAP[args.odr_code]
    filter_name = SINC_FILTER_MAP[args.filter_code]

    logger.info(
        "Step-sine response: %.1f Hz -> %.1f Hz  "
        "(%d pts, %d runs)  ODR=%.0f kHz, Filter=%s, Vpp=%.2f",
        args.freq_start, args.freq_stop,
        args.points, args.runs,
        odr_hz / KILO, filter_name, args.amplitude * 2
    )

    ace.setup_capture(args.samples, args.odr_code)

    freqs = np.logspace(
        math.log10(args.freq_start),
        math.log10(args.freq_stop),
        args.points
    )

    wave_gen = WaveformGenerator(args.sdg_host)
    wave_gen.enable(1)
    wave_gen.enable(2)

    power_runs = np.zeros((args.runs, args.points), dtype=np.float64)

    for current_run in range(args.runs):
        logger.info("Run %d / %d", current_run + 1, args.runs)
        for i, f in enumerate(freqs):
            _, vrms = measure_tone(
                ace, wave_gen, f, odr_hz,
                args.amplitude, args.offset,
                args.settle_cycles, args.capture_cycles,
                logger
            )
            power_runs[current_run, i] = vrms ** 2  # Store power

    wave_gen.disable(1)
    wave_gen.disable(2)

    mean_power = power_runs.mean(axis=0)

    vrms_avg = np.sqrt(mean_power)
    input_peak = args.amplitude
    gains = vrms_avg * np.sqrt(2) / input_peak

    ref = np.median(gains[:max(3, args.points // 20)])  # Median of first 5 %
    # Normalize gains to the reference
    gains_norm = gains / ref
    # Convert to dB
    gdb_norm = 20 * np.log10(gains_norm)

    bandwidth_indices = np.where(gdb_norm <= -3.0)[0]
    if bandwidth_indices.size:
        crossing_idx = bandwidth_indices[0]
        f3db = np.interp(-3.0,
                         [gdb_norm[crossing_idx - 1], gdb_norm[crossing_idx]],
                         [freqs[crossing_idx - 1], freqs[crossing_idx]])
        logger.info(
            "-3 dB bandwidth: %.0f Hz  (mean of %d runs, 95%% CI shown below)",
            f3db, args.runs
        )
    else:
        logger.info("-3 dB point not in sweep range")

    passband_db = gdb_norm[freqs < 1000]  # First decade
    passband_mean = passband_db.mean()
    std = passband_db.std(ddof=1)
    ci95 = 1.96 * std / np.sqrt(len(passband_db))
    logger.info(
        "Pass-band gain: %.2f dB +- %.2f dB (95%% CI), "
        "std = %.2f dB over %d runs",
        passband_mean, ci95, std, args.runs
    )

    plot_freq_response(freqs,
                       gains_norm,
                       runs=args.runs,
                       amplitude_vpp=args.amplitude,
                       out_file=(args.plot if isinstance(args.plot, str)
                                 else "step_bode.png"),
                       show=args.show)


def run_dc_tests(args, logger, ace):
    start_time = time.time()
    next_time_update = start_time + 300  # 5-min heartbeat

    smu = B2912A(args.resource)
    dmm = None if args.no_dmm else Dmm6500Controller(args.dmm_ip)

    dmm_fixed_range = 100
    centre = float(getattr(args, "offset", 0.0))  # Volts
    raw_amp = float(args.amplitude)

    # Ensure the sweep never exceeds the fixed DMM range +/-dmm_fixed_range.
    max_pos_headroom = dmm_fixed_range - centre
    max_neg_headroom = dmm_fixed_range + centre
    safe_amp = 2 * min(max_pos_headroom, max_neg_headroom)
    if safe_amp <= 0:
        raise ValueError(f"Offset {centre} V is outside the +/-{dmm_fixed_range} V "
                         "meter range.")
    amplitude = min(raw_amp, safe_amp)
    voltage_start, voltage_stop = centre - amplitude / 2, centre + amplitude / 2

    if dmm:
        dmm.configure_for_precise_dc(nplc=5,
                                     dmm_range=dmm_fixed_range)
        logger.info("DMM: %d V range, 5 PLC, AZER=ON, rear terminals",
                    dmm_fixed_range)
    else:
        logger.info("--no-dmm - using SMU sensed voltage only")

    logger.info("Sweep: %.3f Vpp centred on %.3f V  "
                "(start %.3f V to stop %.3f V)",
                amplitude, centre, voltage_start, voltage_stop)

    smu.output_on()

    steps = args.steps or DEFAULT_INL_STEPS
    runs = args.runs or DEFAULT_RUNS
    sweep_voltages = np.linspace(voltage_stop, voltage_start, steps)  # descending sweep

    settle_delay_s = 0.05  # 50 ms guard

    if not args.no_board:
        ace.setup_capture(args.samples, args.odr_code)

    run_stats = {"dmm": [], "smu": []}

    try:
        for run_idx in range(1, runs + 1):
            logger.info("Run %d / %d", run_idx, runs)

            actual_v_smu, actual_v_dmm, adc_v = [], [], []

            for steps, voltage in enumerate(sweep_voltages, 1):
                # Command SMU & wait
                smu.set_voltage_blocking(voltage)
                while int(smu.smu.query("STAT:OPER:COND?")) & 0b1:
                    time.sleep(0.002)
                time.sleep(settle_delay_s)  # extra 50 ms

                # SMU self-measurement
                actual_v_smu.append(smu.measure_voltage())

                # DMM reference
                if dmm:
                    _ = dmm.measure_voltage_dc()
                    mean_v, _, _ = dmm.measure_voltage_avg(n_avg=10, delay=0.05)
                    actual_v_dmm.append(mean_v)

                # ADC board or commanded voltage
                if not args.no_board:
                    raw = capture_samples(ace, args.samples, ADC_LSB,
                                        output_dir=os.getcwd())
                    adc_v.append(np.mean(raw))
                else:
                    adc_v.append(voltage)

                # Progress heartbeat
                now = time.time()
                if now >= next_time_update or steps == steps:
                    logger.info("Step %d/%d  (%.4f V)", steps, steps, voltage)
                    next_time_update = now + 300
                print(f"Step {steps}/{steps}: Commanded {voltage:+.6f} V", end="\r", flush=True)

            if not adc_v:
                logger.warning("No data collected; skipping analysis")
                continue

            adc_arr   = np.asarray(adc_v)
            show_plot = getattr(args, "show_plots", False)

            if args.no_board:
                # Source-linearity mode (commanded voltage is X-axis)
                result = analyze_linearity(adc_arr, np.asarray(actual_v_smu), "SMU")
                logger.info(
                    "Run %d [SMU]  Gain=%.6f  Offset=%.1f µV  "
                    "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                    run_idx, result["gain"], result["offset_uV"],
                    result["max_inl_ppm"], result["typ_inl_ppm"]
                )
                if show_plot:
                    plot_dc_linearity_summary(
                        actual_v_run=adc_arr,
                        fit_line_run=result["fit_line"],
                        inl_lsb_run=result["inl_lsb"],
                        runs=1, amplitude_vpp=amplitude, steps=steps,
                        out_file="dc_plot_src_smu.png", show=True
                    )
                run_stats["smu"].append(result)

                if dmm:
                    result = analyze_linearity(adc_arr, np.asarray(actual_v_dmm), "DMM")
                    logger.info(
                        "Run %d [DMM] Gain=%.6f  Offset=%.1f µV  "
                        "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                        run_idx, result["gain"], result["offset_uV"],
                        result["max_inl_ppm"], result["typ_inl_ppm"]
                    )
                    if show_plot:
                        plot_dc_linearity_summary(
                            actual_v_run=adc_arr,
                            fit_line_run=result["fit_line"],
                            inl_lsb_run=result["inl_lsb"],
                            runs=1, amplitude_vpp=amplitude, steps=steps,
                            out_file="dc_plot_src_dmm.png", show=True
                        )
                    run_stats["dmm"].append(result)

            else:
                # Normal ADC INL mode (reference voltage is X-axis)
                result = analyze_linearity(np.asarray(actual_v_smu), adc_arr, "SMU")
                logger.info(
                    "Run %d [SMU]  Gain=%.6f  Offset=%.1f uV  "
                    "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                    run_idx, result["gain"], result["offset_uV"],
                    result["max_inl_ppm"], result["typ_inl_ppm"]
                )
                if show_plot:
                    plot_dc_linearity_summary(
                        actual_v_run=np.asarray(actual_v_smu),
                        fit_line_run=result["fit_line"],
                        inl_lsb_run=result["inl_lsb"],
                        runs=1, amplitude_vpp=amplitude, steps=steps,
                        out_file="dc_plot_smu.png", show=True
                    )
                run_stats["smu"].append(result)

                if dmm:
                    result = analyze_linearity(np.asarray(actual_v_dmm), adc_arr, "DMM")
                    logger.info(
                        "Run %d [DMM]  Gain=%.6f  Offset=%.1f uV  "
                        "Max|INL|=%.2f ppm  Typ|INL|=%.2f ppm",
                        run_idx, result["gain"], result["offset_uV"],
                        result["max_inl_ppm"], result["typ_inl_ppm"]
                    )
                    if show_plot:
                        plot_dc_linearity_summary(
                            actual_v_run=np.asarray(actual_v_dmm),
                            fit_line_run=result["fit_line"],
                            inl_lsb_run=result["inl_lsb"],
                            runs=1, amplitude_vpp=amplitude, steps=steps,
                            out_file="dc_plot_dmm.png", show=True
                        )
                    run_stats["dmm"].append(result)
    finally:
        smu.output_off()
        smu.close()

    for tag in ("dmm", "smu"):
        stats = dc_summary(tag, run_stats)
        if stats is None:
            continue

        logger.info("*** %s reference (avg over %d runs) ***",
                    tag.upper(), stats["runs"])
        logger.info("Gain err      : %.3f %%", stats["gain_err_pct"])
        logger.info("Offset        : %.1f µV", stats["offset_uV"])
        logger.info("Max |INL|     : %.2f ppm FS", stats["max_inl_ppm"])
        logger.info("Typical |INL| : %.2f ppm FS", stats["typ_inl_ppm"])
        logger.info("RMS  INL      : %.3f LSB", stats["rms_inl_lsb"])


# =============================================================================
# Argument-parser construction
# =============================================================================
def add_common_adc_args(parser):
    parser.add_argument('--ace-host', dest='ace_host', type=str,
                        default=ACE_HOST_DEFAULT, help='ACE server address')
    parser.add_argument('--odr-code', type=int, choices=list(SLAVE_ODR_MAP.keys()),
                        default=ADC_ODR_CODE_DEFAULT,
                        help=('ADC ODR code (0-13): ' +
                              ', '.join(f"{c}={r:.0f} Hz"
                                        for c, r in SLAVE_ODR_MAP.items())))
    parser.add_argument('--filter-code', type=int,
                        choices=list(SINC_FILTER_MAP.keys()),
                        default=ADC_FILTER_CODE_DEFAULT,
                        help=('ADC filter code (0-4): ' +
                              ', '.join(f"{c}={name}"
                                        for c, name in SINC_FILTER_MAP.items())))
    parser.add_argument('-n', '--samples', type=int, default=SAMPLES_DEFAULT,
                        help='number of samples to capture')
    parser.add_argument('--runs', type=int, default=5,
                        help='number of repeat measurements')
    parser.add_argument('--adc-channel', type=int, choices=[0, 1, 2, 3], default=1,
                        help='ADC channel to use (0-3)')


def add_common_plot_args(parser):
    parser.add_argument('--plot', action='store_true', help='save plots to files')
    parser.add_argument('--show', action='store_true', help='display plots on screen')


def setup_parsers():
    p = argparse.ArgumentParser(description='EVAL-AD4134 test CLI')
    subs = p.add_subparsers(dest='cmd', required=True)

    # --- Noise floor ---------------------------------------------------------
    nf = subs.add_parser('noise-floor', help='Noise-floor test')
    add_common_adc_args(nf)
    add_common_plot_args(nf)
    nf.add_argument('--histogram', action='store_true', help='plot histogram')
    nf.add_argument('--hist-bins', type=int, default=HIST_BINS_DEFAULT,
                    help='bins for histogram')
    nf.add_argument('--fft', action='store_true', help='perform FFT')

    # --- SFDR / THD / ENOB ---------------------------------------------------
    sf = subs.add_parser('sfdr', help='Dynamic-performance test')
    sf.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip ADC capture')
    sf.add_argument('--freq', type=float, required=True,
                    help='test frequency [Hz]')
    sf.add_argument('--amplitude', type=float, default=DEFAULT_STEP_VPP,
                    help='expected differential Vpp at the ADC inputs')

    add_common_adc_args(sf)
    sf.set_defaults(odr_code=7)
    add_common_plot_args(sf)

    # --- Settling-time -------------------------------------------------------
    st = subs.add_parser('settling-time', help='Transient settling-time test')
    st.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    st.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='SDG channel')
    st.add_argument('--amplitude', type=float, default=DEFAULT_STEP_VPP, help='step Vpp')
    st.add_argument('--offset', type=float, default=0.0, help='offset [V]')
    st.add_argument('--frequency', type=float, default=100.0, help='rep rate [Hz]')
    st.add_argument('--no-board', dest='no_board', action='store_true',
                    help='skip ADC capture (AWG-only verification)')
    st.add_argument('--start-wait', type=float, default=0.5,
                    help='seconds to wait after enabling the AWG before the first capture')
    add_common_adc_args(st)
    add_common_plot_args(st)

    # --- Frequency-response --------------------------------------------------
    fr = subs.add_parser('freq-response', help='Continuous-chirp freq-response gain sweep')
    fr.add_argument('--sdg-host', dest='sdg_host', type=str,
                    default=SDG_HOST_DEFAULT, help='SDG address')
    fr.add_argument('--channel', type=int, choices=[1, 2], default=1,
                    help='AWG channel')
    fr.add_argument('--freq-start', dest='freq_start', type=float, default=400.0,
                    help='Sweep start frequency [Hz]')
    fr.add_argument('--freq-stop', dest='freq_stop', type=float, default=625000.0,
                    help='Sweep stop frequency [Hz]')
    fr.add_argument('--points', dest='points', type=int, default=250,
                    help='Amount of step points to sweep')
    fr.add_argument('--amplitude', dest='amplitude', type=float, default=1.0,
                    help='Differential Vpp of the sine sweep')
    fr.add_argument('--offset', dest='offset', type=float, default=0.0,
                    help='Common-mode DC offset [V]')
    fr.add_argument('--settle_cycles', dest='settle_cycles', type=int, default=8,
                    help='Number of cycles to wait before capturing')
    fr.add_argument('--capture_cycles', dest='capture_cycles', type=int, default=32,
                    help='Number of cycles to capture')
    fr.add_argument('--no-board', dest='no_board', action='store_true',
                    help='Dry-run: skip ADC capture')
    add_common_adc_args(fr)
    add_common_plot_args(fr)

    # --- DC tests ----------------------------------------------------
    dct = subs.add_parser('dc-test', help='DC measurement test (DMM + Source)')
    dct.add_argument('--no-board', dest='no_board', action='store_true',
                     help='Dry-run: skip ADC capture')
    dct.add_argument('--dmm-ip', type=str, default=DMM_IP_DEFAULT,
                     help='DMM IP address (default: %s)' % DMM_IP_DEFAULT)
    dct.add_argument('--resource', type=str,
                     default=f'TCPIP0::{B29_HOST_DEFAULT}::inst0::INSTR',
                     help='B2912A VISA resource')
    dct.add_argument('--amplitude', type=float,
                     default=MAX_INPUT_RANGE * 2,
                     help='Total sweep amplitude [V]. Sweep is from -amplitude/2 to +amplitude/2.')
    dct.add_argument('--steps', type=int, default=4096,
                     help='number of voltage points (default≈2 mV step)')
    dct.add_argument('--no-dmm', dest='no_dmm', action='store_true',
                     help='skip DMM measurements (use ADC only)')
    dct.add_argument('--offset', type=float, default=0.0,
                     help='Common-mode DC offset [V] (default: 0.0 V)')
    add_common_adc_args(dct)
    add_common_plot_args(dct)
    dct.set_defaults(runs=1, samples=4096)

    return p


# =============================================================================
# Main
# =============================================================================
def main():
    args = setup_parsers().parse_args()

    root = 'Measurements'
    os.makedirs(root, exist_ok=True)
    tstamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(root, f"{args.cmd}_{tstamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.chdir(run_dir)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[logging.FileHandler(f"{args.cmd}_results.log"),
                  logging.StreamHandler()]
    )
    logger = logging.getLogger()
    logger.info("Starting '%s' test", args.cmd)

    ace = None
    if not getattr(args, 'no_board', False):
        ace = ACEClient(args.ace_host)
        all_channels = [0, 1, 2, 3]
        disable_list = [str(ch) for ch in all_channels if ch != args.adc_channel]
        disable_channels = ','.join(disable_list)
        ace.configure_board(
            filter_code=args.filter_code,
            disable_channels=disable_channels
        )

    if args.cmd == 'noise-floor':
        run_noise_floor(args, logger, ace)
    elif args.cmd == 'sfdr':
        run_sfdr(args, logger, ace)
    elif args.cmd == 'settling-time':
        run_settling_time(args, logger, ace)
    elif args.cmd == 'freq-response':
        run_freq_response(args, logger, ace)
    elif args.cmd == 'dc-test':
        run_dc_tests(args, logger, ace)
    else:
        logger.error("Unknown command (use -h for help)")


if __name__ == '__main__':
    main()
