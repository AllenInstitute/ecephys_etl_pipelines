import pandas as pd
import argschema

from ecephys_etl.modules.vbn_create_stimulus_table.schemas import (
    VbnCreateStimulusTableInputSchema,
    VbnCreateStimulusTableOutputSchema
)
from ecephys_etl.data_extractors.sync_dataset import Dataset
from ecephys_etl.modules.vbn_create_stimulus_table.create_stim_table import (
    create_vbn_stimulus_table
)


class VbnCreateStimulusTable(argschema.ArgSchemaParser):
    default_schema = VbnCreateStimulusTableInputSchema
    default_output_schema = VbnCreateStimulusTableOutputSchema

    def run(self):
        self.logger.name = type(self).__name__
        output_path = self.args["output_stimulus_table_path"]
        sync_data = Dataset(self.args["sync_h5_path"])
        behavior_data = pd.read_pickle(self.args["behavior_pkl_path"])
        mapping_data = pd.read_pickle(self.args["mapping_pkl_path"])
        replay_data = pd.read_pickle(self.args["replay_pkl_path"])

        stim_table: pd.DataFrame = create_vbn_stimulus_table(
            sync_data=sync_data,
            behavior_data=behavior_data,
            mapping_data=mapping_data,
            replay_data=replay_data
        )
        stim_table.to_csv(path_or_buf=output_path, index=False)

        output_data = {
            "input_parameters": self.args,
            "output_path": output_path
        }

        self.output(output_data, indent=2)


if __name__ == "__main__":  # pragma: nocover
    stim_table_maker = VbnCreateStimulusTable()
    stim_table_maker.run()
