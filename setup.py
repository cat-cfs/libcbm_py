import os
from setuptools import setup
from setuptools import find_packages


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

resources_dir = "resources"

cbm_defaults_db = [
    os.path.join(resources_dir, "cbm_defaults_db", x)
    for x in ["cbm_defaults_v1.2.7739.338.db"]
]

cbm_defaults_queries = [
    os.path.join(resources_dir, "cbm_defaults_queries", "*.sql")
]

win_x86_64_bin = [
    os.path.join(resources_dir, "libcbm_bin", "win_x86_64", x)
    for x in [
        "cbm.lib",
        "core.lib",
        "libcbm.dll",
        "libcbm.exp",
        "libcbm.lib",
        "volume_to_biomass.lib",
    ]
]

ubuntu_binaries = []
for ubuntu_ver in ["ubuntu_18_04_x86_64", "ubuntu_20_04_x86_64"]:
    ubuntu_binaries.extend(
        [
            os.path.join(resources_dir, "libcbm_bin", ubuntu_ver, x)
            for x in ["cbm.a", "core.a", "libcbm.so", "volume_to_biomass.a"]
        ]
    )

mac_os_binaries = []
for mac_os_ver in ["macosx_10_12_x86_64", "macosx_10_15_x86_64"]:
    mac_os_binaries.extend(
        [
            os.path.join(resources_dir, "libcbm_bin", mac_os_ver, x)
            for x in ["cbm.a", "core.a", "libcbm.dylib", "volume_to_biomass.a"]
        ]
    )

test_resources = []
for x in [
    "cbm3_tutorial2",
    "cbm3_tutorial2_eligibilities",
    "cbm3_tutorial6",
    "sit_rule_based_events",
    "moss_c_test_case",
    "moss_c_multiple_stands",
    "sit_spatially_explicit",
]:
    test_resources.append(os.path.join(resources_dir, "test", x, "*.csv"))
    test_resources.append(os.path.join(resources_dir, "test", x, "*.xlsx"))
    test_resources.append(os.path.join(resources_dir, "test", x, "*.json"))


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="libcbm",
    version="2.2.0",
    description="Carbon budget model library based on CBM-CFS3",
    keywords=["cbm-cfs3"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Carbon Accounting Team - Canadian Forest Service",
    author_email="scott.morken@canada.ca",
    maintainer="Scott Morken",
    maintainer_email="scott.morken@canada.ca",
    license="MPL-2.0",
    url="",
    download_url="",
    packages=find_packages(exclude=["test*"]),
    package_data={
        "libcbm": cbm_defaults_db
        + win_x86_64_bin
        + ubuntu_binaries
        + mac_os_binaries
        + cbm_defaults_queries
        + test_resources
    },
    install_requires=requirements,
)
