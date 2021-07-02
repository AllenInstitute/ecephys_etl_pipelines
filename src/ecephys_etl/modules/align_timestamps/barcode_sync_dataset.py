from typing import Tuple
import numpy as np

from ecephys_etl.modules.align_timestamps import barcode
from ecephys_etl.data_extractors.ecephys_sync_dataset import EcephysSyncDataset


class BarcodeSyncDataset(EcephysSyncDataset):

    @property
    def barcode_line(self):
        """ Obtain the index of the barcode line for this dataset.

        Possible line labels for ecephys barcodes can be found in the
        following (internal) MPE sheet:

        https://alleninstitute.sharepoint.com/:x:/s/Instrumentation/ES2bi1xJ3E9NupX-zQeXTlYBS2mVVySycfbCQhsD_jPMUw?e=Z9jCwH
        """
        if "barcode" in self.line_labels:
            return self.line_labels.index("barcode")
        elif "barcodes" in self.line_labels:
            return self.line_labels.index("barcodes")
        elif "barcode_ephys" in self.line_labels:
            return self.line_labels.index("barcode_ephys")
        else:
            raise ValueError("no barcode line found")

    def extract_barcodes(
        self,
        inter_barcode_interval: float = 10.0,
        bar_duration: float = 0.03,
        barcode_duration_ceiling: float = 2.0,
        nbits: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Read barcodes and their times from this dataset's barcode line.

        Parameters
        ----------
        inter_barcode_interval : numeric, optional
            Minimun duration of time between barcodes. By default 10.0
        bar_duration : numeric, optional
            A value slightly shorter than the expected duration of each bar
        barcode_duration_ceiling : numeric, optional
            The maximum duration of a single barcode
        nbits : int, optional
            The bit-depth of each barcode

        Returns
        -------
        times : np.ndarray
            The start times of each detected barcode.
        codes : np.ndarray
            The values of each detected barcode

        """
        sample_freq_digital = float(self.sample_frequency)
        barcode_channel = self.barcode_line

        on_events = self.get_rising_edges(barcode_channel)
        off_events = self.get_falling_edges(barcode_channel)

        on_times = on_events / sample_freq_digital
        off_times = off_events / sample_freq_digital

        return barcode.extract_barcodes_from_times(
            on_times, off_times,
            inter_barcode_interval=inter_barcode_interval,
            bar_duration=bar_duration,
            barcode_duration_ceiling=barcode_duration_ceiling,
            nbits=nbits
        )
