import unittest

from scripts.predict_season import form_gameweek_status


class FormGameweekStatusTests(unittest.TestCase):
    BOOT = {"events": [
        {"id": 3, "finished": True, "data_checked": True},
        {"id": 4, "finished": False, "data_checked": False},
    ]}

    def test_confirmed_gameweek_is_final(self):
        self.assertEqual(form_gameweek_status(self.BOOT, 3),
                         {"finished": True, "data_checked": True})

    def test_provisional_gameweek_is_not(self):
        # 2026-09-15: all GW4 fixtures played, bonus unconfirmed.
        self.assertFalse(form_gameweek_status(self.BOOT, 4)["data_checked"])

    def test_gameweek_zero_has_nothing_provisional(self):
        self.assertTrue(form_gameweek_status(self.BOOT, 0)["data_checked"])


if __name__ == "__main__":
    unittest.main()
