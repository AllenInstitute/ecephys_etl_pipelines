Visual Coding Neuropixels stimulus table creation
=================================================
This module is used to produce the stimulus table for the Visual Coding
Neuropixels (VCN) project.

Builds a table of stimulus parameters. Each row describes a single sweep of
stimulus presentation and has start and end times (in seconds, on the master
clock) as well as the values of each applicable stimulus parameter during
that sweep.

This stimulus table is later used to extract trial window times when
calculating Current Source Density (CSD).

Running
-------
```
python -m ecephys_etl.modules.vcn_create_stimulus_table
--input_json <path to input json> --output_json <path to output json>
```

Input data
----------
See the [schema file](_schemas.py) for detailed information about input json
contents.

Output data
-----------
- Stimulus table csv : The complete stimulus table.
