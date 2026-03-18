# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:36:34 2026
@author: Thomas Webber
"""
# pypamguard_patch.py
# Place this file in the same folder as runClassPGDF.py
# Import it once before loading any pypamguard files.
#
# Fix: pypamguard rounds click waveforms to 4 decimal places (np.round(x, 4))
# giving only ~15 unique amplitude levels per click. PAMGuard uses full double

# This patch removes the rounding to match PAMGuard's behaviour.

import numpy as np
from pypamguard.chunks.modules.detectors.click import ClickDetector
from pypamguard.utils.constants import DTYPES


def _patched_process(self, br, chunk_info):
    # ----------------------------------------------------------------
    # Replicate StandardModule._process (calls super()._process + _process_stddata)
    # ----------------------------------------------------------------
    super(ClickDetector, self)._process(br, chunk_info)

    # ----------------------------------------------------------------
    # Replicate ClickDetector._process with rounding removed
    # ----------------------------------------------------------------
    self.n_chan = len(self.channel_map.get_set_bits())

    if self._module_header.version <= 3:
        self.start_sample, self.channel_map = br.bin_read([DTYPES.INT64, DTYPES.INT32])

    self.trigger_map, self.type = br.bin_read([DTYPES.INT32, DTYPES.INT16])
    self.flags = br.bitmap_read(DTYPES.INT32)

    if self._module_header.version <= 3:
        n_delays = br.bin_read(DTYPES.INT16)
        if n_delays:
            self.delays = br.bin_read(DTYPES.FLOAT32, shape=n_delays)

    n_angles = br.bin_read(DTYPES.INT16)
    if n_angles:
        self.angles = br.bin_read(DTYPES.FLOAT32, shape=n_angles)

    n_angle_errors = br.bin_read(DTYPES.INT16)
    if n_angle_errors:
        self.angle_errors = br.bin_read(DTYPES.FLOAT32, shape=n_angle_errors)

    if self._module_header.version <= 3:
        self.duration = br.bin_read(DTYPES.UINT16)
    else:
        self.duration = self.sample_duration

    max_val = br.bin_read(DTYPES.FLOAT32)

    # Original pypamguard: np.round(x * max_val / 127, 4)  <- rounds to 4dp (~15 levels)
    # PAMGuard Java:       x * max_val / 127               <- full double precision
    def normalize_wave(x):
        return x * max_val / 127

    self.wave = br.bin_read((DTYPES.INT8, normalize_wave), shape=(self.n_chan, self.duration))


# Apply the patch
ClickDetector._process = _patched_process
print("[pypamguard_patch] ClickDetector waveform rounding patch applied.")