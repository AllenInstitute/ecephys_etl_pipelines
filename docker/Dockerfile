FROM continuumio/miniconda3:4.8.2

LABEL maintainer="nicholas.mei@alleninstitute.org"
LABEL version=1.0
LABEL description="This dockerfile provides a working environment for \
                   Allen Institute for Brain Science extracellular \
                   electrophysiology data processing pipelines."

ARG ECEPHYS_ETL_TAG=main

ENV CONDA_ENVS=/envs
ENV ECEPHYS_ETL_ENV=${CONDA_ENVS}/ecephys_etl

RUN mkdir ${CONDA_ENVS}

# NOTE: To install into conda environments during docker build we need to
# use "conda run -n <my_env> subsequent commands". For details see:
# https://pythonspeed.com/articles/activate-conda-dockerfile/

WORKDIR /repos
RUN apt-get -y update \
    && rm -rf /var/lib/apt/* \
    && echo "Using branch ${ECEPHYS_ETL_TAG}"\
    && git clone -b ${ECEPHYS_ETL_TAG} https://github.com/AllenInstitute/ecephys_etl_pipelines ./ecephys_etl \
    && conda create --prefix ${ECEPHYS_ETL_ENV} python=3.8 \
    && conda run --prefix ${ECEPHYS_ETL_ENV} pip install --no-cache ./ecephys_etl \
    && conda run --prefix ${ECEPHYS_ETL_ENV} pip install --no-cache -r ./ecephys_etl/test_requirements.txt \
    && echo "Use for ecephys_etl "$(conda run --prefix ${ECEPHYS_ETL_ENV} which python) \
    && cd /repos \
    && conda clean --all

# leave /repos/ecephys_etl so we can run tests

# The base image has the default entrypoint activate the base conda env.

# If you need to troubleshoot by running interacticely inside the container:
# docker run --rm -it --entrypoint "/bin/bash" alleninstitutepika/ecephys_etl_pipelines

# If you need to pass args that contain quotes:
# docker run --entrypoint /bin/bash --read-only --tmpfs /tmp alleninstitutepika/ecephys_etl_pipelines:<tag> -c "/envs/ecephys_etl/bin/python -m pytest -m 'not suite2p_only'"

# Uses the bash $@ special parameter to consume all docker args after
# image_name:tag as args to container bash shell
ENTRYPOINT ["/bin/bash", "-c", "$@", "--"]
