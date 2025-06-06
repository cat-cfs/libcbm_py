import os
from setuptools import setup
from setuptools import find_packages
from libcbm import __version__ as libcbm_version


this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

resources_dir = "resources"

cbm_defaults_db = [
    os.path.join(resources_dir, "cbm_defaults_db", x)
    for x in ["cbm_defaults_v1.2.8340.362.db"]
]

cbm_defaults_queries = [
    os.path.join(resources_dir, "cbm_defaults_queries", "*.sql")
]

cbm_exn_default_parameters = [
    os.path.join(resources_dir, "cbm_exn", "*.csv"),
    os.path.join(resources_dir, "cbm_exn", "*.json"),
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
for ubuntu_ver in ["ubuntu_22_04_x86_64", "ubuntu_24_04_x86_64"]:
    ubuntu_binaries.extend(
        [
            os.path.join(resources_dir, "libcbm_bin", ubuntu_ver, x)
            for x in ["cbm.a", "core.a", "libcbm.so", "volume_to_biomass.a"]
        ]
    )

mac_os_binaries = []
for mac_os_ver in ["macos_64"]:
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
    "cbm3_tutorial2_extensions",
    "cbm3_tutorial6",
    "sit_rule_based_events",
    "moss_c_test_case",
    "moss_c_multiple_stands",
    "sit_spatially_explicit",
    "cbm_exn_net_increments",
]:
    test_resources.append(os.path.join(resources_dir, "test", x, "*.csv"))
    test_resources.append(os.path.join(resources_dir, "test", x, "*.xlsx"))
    test_resources.append(os.path.join(resources_dir, "test", x, "*.json"))


with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="libcbm",
    version=libcbm_version,
    description="Carbon budget model library based on CBM-CFS3",
    keywords=["cbm-cfs3"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Carbon Accounting Team - Canadian Forest Service",
    maintainer="Scott Morken",
    maintainer_email="scott.morken@nrcan-rncan.gc.ca",
    license="MPL-2.0",
    url="https://github.com/cat-cfs/libcbm_py",
    download_url="",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(exclude=["test*"]),
    package_data={
        "libcbm": cbm_defaults_db
        + win_x86_64_bin
        + ubuntu_binaries
        + mac_os_binaries
        + cbm_defaults_queries
        + cbm_exn_default_parameters
        + test_resources
    },
    install_requires=requirements,
)
