import os
import unittest
from tempfile import NamedTemporaryFile
from unittest.mock import patch
from libcbm import resources


class ResourcesTest(unittest.TestCase):
    def test_get_local_dir(self):
        self.assertTrue(os.path.exists(resources.get_local_dir()))

    def test_get_cbm_defaults_path(self):
        self.assertTrue(os.path.exists(resources.get_cbm_defaults_path()))

    def test_get_test_resources_dir(self):
        self.assertTrue(os.path.exists(resources.get_test_resources_dir()))

    @patch("libcbm.resources.parse_key_value_file")
    @patch("libcbm.resources.os")
    def test_get_linux_os_release(self, _os, parse_key_value_file):
        parse_key_value_file.side_effect = lambda _: "os_release_mock"
        _os.path.exists.side_effect = lambda _: True
        result = resources.get_linux_os_release()
        self.assertTrue(result == "os_release_mock")
        parse_key_value_file.assert_called_once_with("/etc/os-release")

    @patch("libcbm.resources.os")
    def test_get_linux_os_release_no_file(self, _os):
        _os.path.exists.side_effect = lambda _: False
        result = resources.get_linux_os_release()
        self.assertTrue(result is None)

    def test_parse_key_value_file(self):
        mock_os_release_file = [
            "NAME=Fedora",
            'VERSION="17 (Beefy Miracle)"',
            "ID=fedora",
            "VERSION_ID=17",
            'PRETTY_NAME="Fedora 17 (Beefy Miracle)"',
            'ANSI_COLOR="0;34"',
            'CPE_NAME="cpe:/o:fedoraproject:fedora:17"',
            'HOME_URL="https://fedoraproject.org/"',
            'BUG_REPORT_URL="https://bugzilla.redhat.com/"',
        ]

        temp_file = NamedTemporaryFile("w", delete=False)
        temp_file.write(os.linesep.join(mock_os_release_file))
        temp_file.close()

        try:
            result = resources.parse_key_value_file(temp_file.name)

            self.assertTrue(
                result
                == {
                    "NAME": "Fedora",
                    "VERSION": "17 (Beefy Miracle)",
                    "ID": "fedora",
                    "VERSION_ID": "17",
                    "PRETTY_NAME": "Fedora 17 (Beefy Miracle)",
                    "ANSI_COLOR": "0;34",
                    "CPE_NAME": "cpe:/o:fedoraproject:fedora:17",
                    "HOME_URL": "https://fedoraproject.org/",
                    "BUG_REPORT_URL": "https://bugzilla.redhat.com/",
                }
            )
        finally:
            os.unlink(temp_file.name)

    @patch("libcbm.resources.sys")
    def test_fail_on_32_bit_system(self, sys):
        sys.maxsize = 2 ^ 32 - 1
        with self.assertRaises(RuntimeError):
            resources.get_libcbm_bin_path()

    @patch("libcbm.resources.platform")
    def test_get_windows_path(self, platform):
        platform.system.side_effect = lambda: "Windows"
        self.assertTrue(os.path.exists(resources.get_libcbm_bin_path()))

    @patch("libcbm.resources.get_linux_os_release")
    @patch("libcbm.resources.platform")
    def test_linux_no_os_release(self, platform, get_linux_os_release):
        platform.system.side_effect = lambda: "Linux"
        get_linux_os_release.side_effect = lambda: None
        with self.assertRaises(RuntimeError):
            resources.get_libcbm_bin_path()

    @patch("libcbm.resources.get_linux_os_release")
    @patch("libcbm.resources.platform")
    def test_linux_versions(self, platform, get_linux_os_release):
        supported_os_releases = [
            {"NAME": "UBUNTU", "VERSION_ID": "20.04"},
            {"NAME": "Ubuntu", "VERSION_ID": "22.04"},
        ]
        for mock_os_release in supported_os_releases:
            platform.system.side_effect = lambda: "Linux"
            get_linux_os_release.side_effect = lambda: mock_os_release
            self.assertTrue(os.path.exists(resources.get_libcbm_bin_path()))

    @patch("libcbm.resources.get_linux_os_release")
    @patch("libcbm.resources.platform")
    @patch("libcbm.resources.warnings")
    def test_unsupported_linux_versions(
        self, warnings, platform, get_linux_os_release
    ):
        unsupported_os_release = {"NAME": "UNKNOWN", "VERSION_ID": "40.00"}
        platform.system.side_effect = lambda: "Linux"
        get_linux_os_release.side_effect = lambda: unsupported_os_release
        self.assertTrue(os.path.exists(resources.get_libcbm_bin_path()))
        warnings.warn.assert_called_once()

    @patch("libcbm.resources.platform")
    def test_mac_os_supported_vers(self, platform):
        supported_vers = ["10.12.xx", "10.13.xx", "10.14.xx", "10.15.xx"]
        platform.system.side_effect = lambda: "Darwin"
        for supported_ver in supported_vers:
            platform.mac_ver.side_effect = [[supported_ver]]
            self.assertTrue(os.path.exists(resources.get_libcbm_bin_path()))

    @patch("libcbm.resources.platform")
    def test_mac_os_unsupported_vers(self, platform):
        supported_vers = ["9.01.xx", "10.11.xx", "13.xx"]
        platform.system.side_effect = lambda: "Darwin"
        for supported_ver in supported_vers:
            platform.mac_ver.side_effect = [[supported_ver]]
            with self.assertRaises(RuntimeError):
                resources.get_libcbm_bin_path()

    @patch("libcbm.resources.platform")
    def test_unsupported_platform(self, platform):
        platform.system.side_effect = lambda: "Java"
        with self.assertRaises(RuntimeError):
            resources.get_libcbm_bin_path()
