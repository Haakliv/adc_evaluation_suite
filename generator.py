from zolve_instruments import Sdg6022x
from ace_client import limit_vpp_offset, MAX_INPUT_RANGE

# Wrapper for Siglent SDG6022X to configure and control output waveforms. Supports sine and pulse with safe amplitude limiting.
class WaveformGenerator:
    offset = 0.0

    def __init__(self, host='192.168.1.100', waveform='SINE', offset=0.0):
        max_in = MAX_INPUT_RANGE*2

        if abs(offset) > max_in:
            offset = max_in if offset > 0 else -max_in
        self.offset = offset

        self.sdg = Sdg6022x(host)
        self.disable(1)
        self.disable(2)
        self.sdg.interface.write("C1:MODE PHASE-LOCKED")
        self.sdg.interface.write("C2:OUTP PLRT,INVT")
        self.sdg.set_waveform(waveform, 1)
        self.sdg.set_waveform(waveform, 2)
        self.sdg.set_offset(offset, 1)
        self.sdg.set_offset(offset, 2)

    def disable(self, channel):
        self.sdg.disable_output(channel)

    def enable(self, channel):
        self.sdg.enable_output(channel)

    def enable_trigger_out(self, channel: int = 1):
        """Enable trigger output on the specified channel."""
        self.sdg.interface.write(f"C{channel}:OUTP TRMD,ON")

    def disable_trigger_out(self, channel: int = 1):
        """Disable trigger output on the specified channel."""
        self.sdg.interface.write(f"C{channel}:OUTP TRMD,OFF")

    def set_trigger_mode(self, mode: str = "RISE", channel: int = 1):
        """
        Set trigger mode for the channel.
        
        Args:
            mode: Trigger mode - "OFF", "RISE", "FALL", "EDGE", "PULS", or "BURST"
            channel: Channel number (1 or 2)
        """
        valid_modes = ["OFF", "RISE", "FALL", "EDGE", "PULS", "BURST"]
        if mode.upper() not in valid_modes:
            raise ValueError(f"Invalid trigger mode. Must be one of {valid_modes}")
        self.sdg.interface.write(f"C{channel}:BTWV TRMD,{mode.upper()}")

    def set_burst_mode(self, ncycles: int = 1, channel: int = 1, trigger_source: str = "EXT"):
        """
        Configure burst mode for the channel.
        
        Args:
            ncycles: Number of cycles per burst (1-1000000)
            channel: Channel number (1 or 2)
            trigger_source: Trigger source - "EXT" (external), "INT" (internal), or "MAN" (manual)
        """
        valid_sources = ["EXT", "INT", "MAN"]
        if trigger_source.upper() not in valid_sources:
            raise ValueError(f"Invalid trigger source. Must be one of {valid_sources}")
        
        # Set burst mode
        self.sdg.interface.write(f"C{channel}:BTWV STATE,ON")
        self.sdg.interface.write(f"C{channel}:BTWV TRSR,{trigger_source.upper()}")
        self.sdg.interface.write(f"C{channel}:BTWV TIME,{ncycles}")
        self.sdg.interface.write(f"C{channel}:BTWV DLAY,0")

    def disable_burst_mode(self, channel: int = 1):
        """Disable burst mode for the channel."""
        self.sdg.interface.write(f"C{channel}:BTWV STATE,OFF")

    def pulse_diff(
        self,
        frequency: float,
        amplitude: float,
        low_percent: float = 80.0, # 4 ms low, 1 ms high at 200 Hz
        edge_time: float = 2e-9,
        ch_pos: int = 1,
        ch_neg: int = 2,
        enable_trigger_out: bool = False,
    ):

        safe_vpp = limit_vpp_offset(amplitude, self.offset)
        if safe_vpp <= 0:
            raise ValueError("Offset too close to rail for given Vpp")

        duty_cycle_percent = 100.0 - low_percent  # SDG’s DUTY = %HIGH

        for ch in (ch_pos, ch_neg):
            self.sdg.set_frequency(frequency, ch)
            self.sdg.set_amplitude(safe_vpp, ch)

            # 50 ohm load
            self.sdg.interface.write(f"C{ch}:OUTP LOAD,50")

            self.sdg.interface.write(f"C{ch}:BSWV DUTY,{duty_cycle_percent}")
            self.sdg.interface.write(f"C{ch}:BSWV DLY,0")
            self.sdg.interface.write(f"C{ch}:BSWV RISE,{edge_time}")
            self.sdg.interface.write(f"C{ch}:BSWV FALL,{edge_time}")

        self.sdg.interface.write(f"C{ch_neg}:OUTP PLRT,INVT")

        # Enable trigger output if requested (typically on ch_pos)
        if enable_trigger_out:
            self.enable_trigger_out(ch_pos)

        self.sdg.enable_output(ch_pos)
        self.sdg.enable_output(ch_neg)

    def sine_diff(self,
                  frequency: float,
                  amplitude: float,
                  offset: float,
                  ch_pos: int = 1,
                  ch_neg: int = 2):
        max_in = MAX_INPUT_RANGE*2
        if abs(offset) > max_in:
            offset = max_in if offset > 0 else -max_in

        safe_vpp = limit_vpp_offset(amplitude, offset)
        if safe_vpp <= 0:
            raise ValueError(
                f"Offset={offset} V leaves no headroom for Vpp={amplitude} V"
            )

        self.sdg.set_frequency(frequency, ch_pos)
        self.sdg.set_frequency(frequency, ch_neg)
        self.sdg.set_amplitude(safe_vpp, ch_pos)
        self.sdg.set_amplitude(safe_vpp, ch_neg)
