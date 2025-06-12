import unittest
import os
from src.rbac_config import (
    get_allowed_documents,
    ALL_FILES,
    FINANCE_FILES,
    MARKETING_FILES,
    HR_FILES,
    ENGINEERING_FILES,
    GENERAL_FILES,
    BASE_PATH
)

class TestRBACConfig(unittest.TestCase):

    def test_finance_team_permissions(self):
        expected_docs = FINANCE_FILES + [
            os.path.join(BASE_PATH, "marketing/marketing_report_2024.md"),
            os.path.join(BASE_PATH, "general/employee_handbook.md")
        ]
        # Ensure all paths are normalized for comparison if needed, though os.path.join should be consistent.
        # Using set for comparison is robust against order differences and duplicates.
        self.assertCountEqual(get_allowed_documents("finance_team"), expected_docs)

    def test_marketing_team_permissions(self):
        self.assertCountEqual(get_allowed_documents("marketing_team"), MARKETING_FILES)

    def test_hr_team_permissions(self):
        self.assertCountEqual(get_allowed_documents("hr_team"), HR_FILES + GENERAL_FILES)

    def test_engineering_department_permissions(self):
        self.assertCountEqual(get_allowed_documents("engineering_department"), ENGINEERING_FILES)

    def test_c_level_executives_permissions(self):
        self.assertCountEqual(get_allowed_documents("c_level_executives"), ALL_FILES)

    def test_employee_level_permissions(self):
        self.assertCountEqual(get_allowed_documents("employee_level"), GENERAL_FILES)

    def test_unknown_role_permissions(self):
        self.assertEqual(get_allowed_documents("unknown_role"), [])

    def test_all_defined_roles_covered(self):
        # A meta-test to ensure all roles in ROLE_PERMISSIONS are tested.
        # This requires ROLE_PERMISSIONS to be imported if not already.
        from src.rbac_config import ROLE_PERMISSIONS
        defined_roles = ROLE_PERMISSIONS.keys()
        tested_roles = [
            "finance_team", "marketing_team", "hr_team",
            "engineering_department", "c_level_executives", "employee_level"
        ]
        self.assertCountEqual(defined_roles, tested_roles, "Not all defined roles are covered in tests.")

if __name__ == '__main__':
    # This allows running the tests directly from this file: python src/tests/test_rbac_config.py
    unittest.main()
