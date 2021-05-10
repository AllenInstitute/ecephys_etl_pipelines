from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
      name="ecephys_etl_pipelines",
      use_scm_version=True,
      description=("Pipelines and transforms for processing extracellular "
                   "electrophysiology data."),
      author=("Dan Kapner, Nicholas Mei, Scott Daniel, "
              "Adam Amster, Taylor Matyasz"),
      author_email="nicholas.mei@alleninstitute.org",
      url="https://github.com/AllenInstitute/ecephys_etl_pipelines",
      package_dir={"": "src"},
      packages=find_packages(where="src"),
      setup_requires=["setuptools_scm"],
      install_requires=required,
)
