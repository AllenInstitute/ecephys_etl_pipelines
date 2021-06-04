Visual Behavior Neuropixels stimulus table creation
===================================================
This module is used to produce the stimulus table for the Visual Behavior
Neuropixels (VBN) project.

Builds a table of stimulus parameters. Each row describes a single sweep of
stimulus presentation and has start and end times (in seconds, on the master
clock) as well as the values of each applicable stimulus parameter during
that sweep.

This stimulus table is later used to extract trial window times when
calculating Current Source Density (CSD).

Running
-------
```
python -m ecephys_etl.modules.vbn_create_stimulus_table
--input_json <path to input json> --output_json <path to output json>
```

Input data
----------

See the [schema file](schemas.py) for detailed information about expected
input parameters.

Output data
-----------
- Stimulus table csv : A stimulus table containing presentations from
    behavior, mapping, and replay pkls. More details about the table can be
    viewed at:

    http://confluence.corp.alleninstitute.org/display/IT/Visual+Behavior+Neuropixels+-+Stimulus+Table+Design+Document
