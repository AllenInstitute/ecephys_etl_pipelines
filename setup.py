from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

with open("test_requirements.txt", "r") as f:
    test_requirements = f.read().splitlines()

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
    python_requires=">=3.8",
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    extras_requires={
        "test": test_requirements
    }
)
