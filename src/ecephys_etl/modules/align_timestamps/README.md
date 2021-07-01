Align Timestamps
================
Computes a transformation from probe sample indices to times on the experiment master clock, then maps zero or more
timestamp arrays through that transform.

Running
-------
```
python -m ecephys_etl.modules.align_timestamps --input_json <path to input json> --output_json <path to output json>
```
See the schema file for detailed information about input json contents.

Input data
----------
- Sync h5 : Contains information about barcode pulses assessed on the master clock
- For each probe
    - barcode channel states file: lists rising and falling edges on the probe's barcode line
    - barcode timestamps file: lists probe samples at which rising and falling edges were detected
    - mappable timestamp files: Will be transformed to the master clock. Some examples:
        - A file listing timestamps of detected spikes
        - A file listing timestamps of detected LFPs

Output data
-----------
Each mappable file (e.g. file containing spikes and/or LFP timestamps) for each probe is aligned to the experiment
master clock and written out. Additionally, the transform (total_time_shift) for aligning each mappable timeseries
file is written into the output json. 
