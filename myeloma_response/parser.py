"""
Excel parser for multiple myeloma lab data.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import re


@dataclass
class LabData:
    """Container for parsed lab data."""
    dates: list[datetime]
    spep: list[Optional[float]]  # M-protein values
    kappa: list[Optional[float]]  # Serum Kappa values
    lambda_: list[Optional[float]]  # Serum Lambda values
    upep: list[Optional[float]]  # Urine M-protein values (optional)

    def __len__(self):
        return len(self.dates)

    def get_baseline_values(self) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """Get the first non-null values for SPEP, Kappa, Lambda."""
        spep = next((v for v in self.spep if v is not None and not np.isnan(v)), None)
        kappa = next((v for v in self.kappa if v is not None and not np.isnan(v)), None)
        lambda_ = next((v for v in self.lambda_ if v is not None and not np.isnan(v)), None)
        return spep, kappa, lambda_


class ExcelParser:
    """Parser for multiple myeloma lab data Excel files."""

    # Reference value patterns to identify test types
    KAPPA_REF = "3.30~19.40"
    LAMBDA_REF = "5.71~26.30"

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw_df: Optional[pd.DataFrame] = None

    def parse(self) -> LabData:
        """Parse the Excel file and return structured lab data."""
        # Read raw Excel data
        self.raw_df = pd.read_excel(self.file_path, header=None)

        # Find the header row (contains 'No', '검사항목', etc.)
        header_row_idx = self._find_header_row()

        # Find data rows for each test
        spep_row, kappa_row, lambda_row, upep_row = self._find_test_rows(header_row_idx)

        # Find date columns (starting from column 5)
        dates = self._parse_dates(header_row_idx)

        # Extract values for each test
        spep_values = self._extract_values(spep_row, len(dates))
        kappa_values = self._extract_values(kappa_row, len(dates))
        lambda_values = self._extract_values(lambda_row, len(dates))
        upep_values = self._extract_values(upep_row, len(dates)) if upep_row is not None else [None] * len(dates)

        return LabData(
            dates=dates,
            spep=spep_values,
            kappa=kappa_values,
            lambda_=lambda_values,
            upep=upep_values
        )

    def _find_header_row(self) -> int:
        """Find the row index containing the headers."""
        for idx, row in self.raw_df.iterrows():
            if 'No' in row.values or '검사항목' in row.values:
                return idx
        raise ValueError("Could not find header row in Excel file")

    def _find_test_rows(self, header_row_idx: int) -> tuple[int, int, int, Optional[int]]:
        """Find row indices for each test type based on reference values or test names."""
        spep_row = None
        kappa_row = None
        lambda_row = None
        upep_row = None

        # Look for test rows after the header
        for idx in range(header_row_idx + 1, min(header_row_idx + 10, len(self.raw_df))):
            row = self.raw_df.iloc[idx]
            row_values = [str(v).strip() if pd.notna(v) else "" for v in row.values]

            # Check for SPEP (Monoclonal peak in Serum)
            if any("SPEP" in v.upper() for v in row_values):
                if any("Serum" in v for v in row_values):
                    spep_row = idx
                elif any("urine" in v.lower() for v in row_values):
                    upep_row = idx
                elif spep_row is None:  # First SPEP found
                    spep_row = idx
                else:  # Second SPEP is likely UPEP
                    upep_row = idx

            # Check for UPEP
            if any("UPEP" in v.upper() for v in row_values):
                upep_row = idx

            # Check for Kappa/Lambda using reference values (more reliable)
            ref_col = 4  # Reference value column
            if len(row) > ref_col:
                ref_value = str(row.iloc[ref_col]).strip() if pd.notna(row.iloc[ref_col]) else ""

                if self.KAPPA_REF in ref_value:
                    kappa_row = idx
                elif self.LAMBDA_REF in ref_value:
                    lambda_row = idx

            # Fallback: check by test description
            for v in row_values:
                if "Kappa light chain" in v:
                    if kappa_row is None:
                        kappa_row = idx
                elif "Lambda light chain" in v:
                    if lambda_row is None:
                        lambda_row = idx

        if spep_row is None:
            raise ValueError("Could not find SPEP row in Excel file")
        if kappa_row is None:
            raise ValueError("Could not find Kappa row in Excel file")
        if lambda_row is None:
            raise ValueError("Could not find Lambda row in Excel file")

        return spep_row, kappa_row, lambda_row, upep_row

    def _parse_dates(self, header_row_idx: int) -> list[datetime]:
        """Parse dates from the header row."""
        dates = []
        header_row = self.raw_df.iloc[header_row_idx]

        # Date columns start from column 5
        for col_idx in range(5, len(header_row)):
            cell_value = header_row.iloc[col_idx]
            if pd.isna(cell_value):
                continue

            date_str = str(cell_value).strip()
            parsed_date = self._parse_date_string(date_str)
            if parsed_date:
                dates.append(parsed_date)

        return dates

    def _parse_date_string(self, date_str: str) -> Optional[datetime]:
        """Parse various date string formats."""
        # Remove newlines and extra whitespace
        date_str = re.sub(r'\s+', ' ', date_str.replace('\n', ' ')).strip()

        # Try different date formats
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%Y- %m- %d %H:%M",  # With spaces
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # Try parsing with regex for format like "2019- 09-04 11:03"
        match = re.search(r'(\d{4})-?\s*(\d{2})-?\s*(\d{2})\s*(\d{2}):(\d{2})', date_str)
        if match:
            year, month, day, hour, minute = match.groups()
            return datetime(int(year), int(month), int(day), int(hour), int(minute))

        return None

    def _extract_values(self, row_idx: int, num_dates: int) -> list[Optional[float]]:
        """Extract numeric values from a data row."""
        values = []
        row = self.raw_df.iloc[row_idx]

        # Data columns start from column 5
        for col_idx in range(5, 5 + num_dates):
            if col_idx >= len(row):
                values.append(None)
                continue

            cell_value = row.iloc[col_idx]
            if pd.isna(cell_value):
                values.append(None)
            else:
                try:
                    values.append(float(cell_value))
                except (ValueError, TypeError):
                    values.append(None)

        return values
