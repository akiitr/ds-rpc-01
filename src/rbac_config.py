import os

BASE_PATH = "resources/data"

ENGINEERING_FILES = [
    os.path.join(BASE_PATH, "engineering/engineering_master_doc.md"),
]

FINANCE_FILES = [
    os.path.join(BASE_PATH, "finance/financial_summary.md"),
    os.path.join(BASE_PATH, "finance/quarterly_financial_report.md"),
]

GENERAL_FILES = [
    os.path.join(BASE_PATH, "general/employee_handbook.md"),
]

HR_FILES = [
    os.path.join(BASE_PATH, "hr/hr_data.csv"),
]

MARKETING_FILES = [
    os.path.join(BASE_PATH, "marketing/market_report_q4_2024.md"),
    os.path.join(BASE_PATH, "marketing/marketing_report_2024.md"),
    os.path.join(BASE_PATH, "marketing/marketing_report_q1_2024.md"),
    os.path.join(BASE_PATH, "marketing/marketing_report_q2_2024.md"),
    os.path.join(BASE_PATH, "marketing/marketing_report_q3_2024.md"),
]

ALL_FILES = ENGINEERING_FILES + FINANCE_FILES + GENERAL_FILES + HR_FILES + MARKETING_FILES

ROLE_PERMISSIONS = {
    "finance_team": FINANCE_FILES + [
        os.path.join(BASE_PATH, "marketing/marketing_report_2024.md"),
        os.path.join(BASE_PATH, "general/employee_handbook.md"),
    ],
    "marketing_team": MARKETING_FILES,
    "hr_team": HR_FILES + GENERAL_FILES,
    "engineering_department": ENGINEERING_FILES,
    "c_level_executives": ALL_FILES,
    "employee_level": GENERAL_FILES,
}

def get_allowed_documents(role: str) -> list[str]:
    """
    Returns a list of allowed document paths for a given role.

    Args:
        role: The role string.

    Returns:
        A list of allowed document paths, or an empty list if the role is not found.
    """
    return ROLE_PERMISSIONS.get(role, [])
