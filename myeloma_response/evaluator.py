"""
Response evaluator for multiple myeloma treatment.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import math

from .classifier import PatientType, ClassificationResult
from .parser import LabData


class ResponseType(Enum):
    """Treatment response categories for multiple myeloma."""
    CR = "bCR"  # Biochemical Complete Response
    VGPR = "VGPR"  # Very Good Partial Response
    PR = "PR"  # Partial Response
    MR = "MR"  # Minor Response
    SD = "SD"  # Stable Disease
    PROGRESSION = "Progression"
    PROGRESSION_TYPE_CHANGE = "Progression (Type 변경 가능!)"  # Progression with possible type change
    LCD_TYPE_CHECK = "LCD Type 변경 확인 필요!"  # FLC difference > 100 in IgG patient
    NOT_EVALUABLE = "NE"  # Not Evaluable

    def __str__(self):
        return self.value


@dataclass
class TimePointResult:
    """Response evaluation result for a single time point."""
    date: datetime
    spep: Optional[float]
    kappa: Optional[float]
    lambda_: Optional[float]
    upep: Optional[float]
    percent_change_from_baseline: Optional[float]
    change_from_nadir: Optional[float]
    nadir_value: Optional[float]
    current_response: Optional[ResponseType]
    confirmed_response: Optional[ResponseType]
    notes: str = ""


@dataclass
class EvaluationResult:
    """Complete evaluation result for a patient."""
    patient_type: PatientType
    classification_reason: str
    baseline_spep: Optional[float]
    baseline_kappa: Optional[float]
    baseline_lambda: Optional[float]
    timepoints: list[TimePointResult] = field(default_factory=list)


class ResponseEvaluator:
    """
    Evaluator for multiple myeloma treatment response.
    (Modified IMWG criteria for real-world data)

    For IgG type patients (SPEP >= 0.5):
    - Response is based on M-protein (SPEP) reduction from baseline
    - MR: >= 25% reduction
    - PR: >= 50% reduction
    - VGPR: >= 90% reduction
    - bCR: SPEP = 0 (biochemical Complete Response)
    - PD: >= 25% increase from nadir AND >= 0.5 g/dL absolute increase
    - LCD Type Check: If |Kappa-Lambda| > 100 at any time → "LCD Type 변경 확인 필요!"

    For LCD type patients (SPEP < 0.5, |Kappa-Lambda| > 100):
    - Progression (Type Change): SPEP >= 0.5 (possible change to IgG type)
    - bCR: FLC ratio (Kappa/Lambda) in normal range (0.26~1.65)
    - VGPR: iFLC >= 90% decrease from baseline OR iFLC < 100
    - PR: iFLC >= 50% decrease from baseline
    - PD: iFLC >= 25% increase from nadir AND absolute increase >= 100 from nadir
    - Note: If only 25% increase → SD with "다른 증상 확인 필요!"

    Confirmation requires 2 consecutive identical responses.
    """

    # IgG Response thresholds (percentage reduction from baseline)
    MR_THRESHOLD = 25.0
    PR_THRESHOLD = 50.0
    VGPR_THRESHOLD = 90.0

    # IgG Progression thresholds (both must be met)
    IGG_PD_PERCENT_THRESHOLD = 25.0  # >= 25% increase from nadir
    IGG_PD_ABSOLUTE_THRESHOLD = 0.5  # >= 0.5 g/dL absolute increase from nadir

    # LCD Response thresholds
    LCD_VGPR_THRESHOLD = 90.0  # 90% decrease
    LCD_PR_THRESHOLD = 50.0    # 50% decrease
    LCD_PD_PERCENT_THRESHOLD = 25.0  # 25% increase
    LCD_PD_ABSOLUTE_THRESHOLD = 100  # Absolute increase >= 100
    LCD_VGPR_ABSOLUTE_THRESHOLD = 100  # iFLC < 100 for VGPR

    # FLC ratio normal range
    FLC_RATIO_LOW = 0.26
    FLC_RATIO_HIGH = 1.65

    # Type change threshold
    TYPE_CHANGE_FLC_DIFF = 100  # |Kappa - Lambda| > 100
    SPEP_THRESHOLD = 0.5  # g/dL - for LCD type change detection

    # Value combination window
    COMBINATION_WINDOW_DAYS = 3  # Combine values within 3 days

    def evaluate(
        self,
        lab_data: LabData,
        classification: ClassificationResult
    ) -> EvaluationResult:
        """
        Evaluate treatment response for a patient.

        Args:
            lab_data: Parsed laboratory data
            classification: Patient classification result

        Returns:
            EvaluationResult with all timepoint evaluations
        """
        result = EvaluationResult(
            patient_type=classification.patient_type,
            classification_reason=classification.classification_reason,
            baseline_spep=classification.baseline_spep,
            baseline_kappa=classification.baseline_kappa,
            baseline_lambda=classification.baseline_lambda
        )

        if classification.patient_type.is_igg_type():
            result.timepoints = self._evaluate_igg_type(lab_data, classification)
        elif classification.patient_type.is_lcd_type():
            result.timepoints = self._evaluate_lcd_type(lab_data, classification)
        else:
            result.timepoints = self._evaluate_unclassified(lab_data)

        return result

    def _check_flc_ratio_normal(self, kappa: Optional[float], lambda_: Optional[float]) -> bool:
        """Check if FLC ratio is in normal range (0.26~1.65)."""
        if kappa is None or lambda_ is None or lambda_ == 0:
            return False
        ratio = kappa / lambda_
        return self.FLC_RATIO_LOW <= ratio <= self.FLC_RATIO_HIGH

    def _check_type_change_possible(self, kappa: Optional[float], lambda_: Optional[float]) -> bool:
        """Check if type change to LCD is possible (|Kappa - Lambda| > 100)."""
        if kappa is None or lambda_ is None:
            return False
        return abs(kappa - lambda_) > self.TYPE_CHANGE_FLC_DIFF

    def _is_value_valid(self, value: Optional[float]) -> bool:
        """Check if a value is valid (not None and not NaN)."""
        if value is None:
            return False
        if isinstance(value, float) and math.isnan(value):
            return False
        return True

    def _get_combined_values(
        self,
        lab_data: LabData,
        current_idx: int,
        combined_indices: set[int]
    ) -> tuple[Optional[float], Optional[float], Optional[float], list[int], str]:
        """
        Get combined SPEP, Kappa, Lambda values from within 3 days.

        Looks back from current_idx to find values within COMBINATION_WINDOW_DAYS.
        Returns the combined values and indices that were used.

        Args:
            lab_data: Laboratory data
            current_idx: Current timepoint index
            combined_indices: Set of indices already used in combinations

        Returns:
            Tuple of (spep, kappa, lambda_, used_indices, combination_note)
        """
        current_date = lab_data.dates[current_idx]
        window_start = current_date - timedelta(days=self.COMBINATION_WINDOW_DAYS)

        # Start with current values
        spep = lab_data.spep[current_idx]
        kappa = lab_data.kappa[current_idx]
        lambda_ = lab_data.lambda_[current_idx]

        used_indices = [current_idx]
        value_sources = {}  # Track which date each value came from

        # Track current values
        if self._is_value_valid(spep):
            value_sources['SPEP'] = current_date
        if self._is_value_valid(kappa):
            value_sources['Kappa'] = current_date
        if self._is_value_valid(lambda_):
            value_sources['Lambda'] = current_date

        # Look back for missing values
        for j in range(current_idx - 1, -1, -1):
            prev_date = lab_data.dates[j]

            # Stop if outside the window
            if prev_date < window_start:
                break

            # Skip if this index is already used in another combination
            if j in combined_indices:
                continue

            # Try to fill missing values
            if not self._is_value_valid(spep) and self._is_value_valid(lab_data.spep[j]):
                spep = lab_data.spep[j]
                used_indices.append(j)
                value_sources['SPEP'] = prev_date

            if not self._is_value_valid(kappa) and self._is_value_valid(lab_data.kappa[j]):
                kappa = lab_data.kappa[j]
                if j not in used_indices:
                    used_indices.append(j)
                value_sources['Kappa'] = prev_date

            if not self._is_value_valid(lambda_) and self._is_value_valid(lab_data.lambda_[j]):
                lambda_ = lab_data.lambda_[j]
                if j not in used_indices:
                    used_indices.append(j)
                value_sources['Lambda'] = prev_date

        # Build combination note if values came from different dates
        combination_note = ""
        if len(set(value_sources.values())) > 1:
            # Values came from multiple dates
            sources = []
            for key, date in sorted(value_sources.items(), key=lambda x: x[1]):
                if date != current_date:
                    sources.append(f"{key}:{date.strftime('%m/%d')}")
            if sources:
                combination_note = f"[결합: {', '.join(sources)}]"

        return spep, kappa, lambda_, used_indices, combination_note

    def _evaluate_igg_type(
        self,
        lab_data: LabData,
        classification: ClassificationResult
    ) -> list[TimePointResult]:
        """Evaluate IgG type patients based on M-protein (SPEP)."""
        timepoints = []
        baseline = classification.baseline_spep

        if baseline is None or baseline == 0:
            # Cannot evaluate without valid baseline
            for i in range(len(lab_data)):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=lab_data.spep[i],
                    kappa=lab_data.kappa[i],
                    lambda_=lab_data.lambda_[i],
                    upep=lab_data.upep[i] if lab_data.upep else None,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=None,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=None,
                    notes="Invalid baseline SPEP value"
                ))
            return timepoints

        # Track nadir (minimum value) for progression detection
        nadir = baseline
        previous_responses: list[ResponseType] = []
        confirmed_response: Optional[ResponseType] = None
        combined_indices: set[int] = set()  # Track indices used in combinations

        for i in range(len(lab_data)):
            raw_spep = lab_data.spep[i]
            raw_kappa = lab_data.kappa[i]
            raw_lambda = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Check if this index was already combined into a later evaluation
            if i in combined_indices:
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=None,
                    confirmed_response=confirmed_response,
                    notes="→ 다음 행에서 결합 평가됨"
                ))
                continue

            # Try to get combined values within 3-day window
            spep, kappa, lambda_, used_indices, combination_note = self._get_combined_values(
                lab_data, i, combined_indices
            )

            # Mark used indices (except current) as combined and update their timepoint entries
            for idx in used_indices:
                if idx != i:
                    combined_indices.add(idx)
                    # Update the already-added timepoint to show it was combined
                    if idx < len(timepoints):
                        old_tp = timepoints[idx]
                        timepoints[idx] = TimePointResult(
                            date=old_tp.date,
                            spep=old_tp.spep,
                            kappa=old_tp.kappa,
                            lambda_=old_tp.lambda_,
                            upep=old_tp.upep,
                            percent_change_from_baseline=None,
                            change_from_nadir=None,
                            nadir_value=old_tp.nadir_value,
                            current_response=None,
                            confirmed_response=old_tp.confirmed_response,
                            notes="→ 다음 행에서 결합 평가됨"
                        )

            # Skip if SPEP is still missing after combination
            if not self._is_value_valid(spep):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response,
                    notes="Missing SPEP value"
                ))
                continue

            # Calculate percent change from baseline
            percent_change = ((baseline - spep) / baseline) * 100

            # Calculate change from nadir
            change_from_nadir = spep - nadir
            percent_increase_from_nadir = ((spep - nadir) / nadir * 100) if nadir > 0 else 0

            # Determine current response
            current_response, response_notes = self._determine_igg_response(
                spep, percent_change, change_from_nadir, percent_increase_from_nadir, kappa, lambda_
            )

            # Track if bCR was achieved
            if current_response == ResponseType.CR:
                cr_achieved = True

            # Add combination note if values were combined
            notes = response_notes
            if combination_note:
                notes = f"{combination_note} {notes}".strip()

            # Update nadir if current value is lower
            if spep < nadir:
                nadir = spep

            # Determine confirmed response - update when 2 consecutive same responses occur
            if previous_responses and previous_responses[-1] == current_response:
                # 2 consecutive same responses -> confirm this response
                confirmed_response = current_response
                # Preserve LCD warning if present
                lcd_warn = " (LCD Type 변경 확인!)" if "(LCD Type 변경 확인!)" in response_notes else ""
                notes = f"{current_response.value} confirmed{lcd_warn}"
                if combination_note:
                    notes = f"{combination_note} {notes}"

            timepoints.append(TimePointResult(
                date=lab_data.dates[i],
                spep=spep,
                kappa=kappa,
                lambda_=lambda_,
                upep=upep,
                percent_change_from_baseline=percent_change,
                change_from_nadir=change_from_nadir,
                nadir_value=nadir,
                current_response=current_response,
                confirmed_response=confirmed_response,
                notes=notes
            ))

            previous_responses.append(current_response)

        return timepoints

    def _determine_igg_response(
        self,
        spep: float,
        percent_change: float,
        change_from_nadir: float,
        percent_increase_from_nadir: float,
        kappa: Optional[float] = None,
        lambda_: Optional[float] = None
    ) -> tuple[ResponseType, str]:
        """
        Determine response type for IgG patients based on SPEP value.

        Returns:
            Tuple of (ResponseType, notes string)
        """
        # Check if LCD type change warning is needed
        lcd_warning = ""
        if self._check_type_change_possible(kappa, lambda_):
            lcd_warning = " (LCD Type 변경 확인!)"

        # Check for bCR first
        if spep == 0:
            return ResponseType.CR, f"SPEP = 0{lcd_warning}"

        # Check for Progression (>= 25% increase AND >= 0.5 g/dL absolute from nadir)
        has_percent_increase = percent_increase_from_nadir >= self.IGG_PD_PERCENT_THRESHOLD
        has_absolute_increase = change_from_nadir >= self.IGG_PD_ABSOLUTE_THRESHOLD

        if has_percent_increase and has_absolute_increase:
            return ResponseType.PROGRESSION, f"SPEP {percent_increase_from_nadir:.1f}% 증가 from nadir (절대 증가 {change_from_nadir:.2f}){lcd_warning}"

        # Check response depth based on percent change from baseline
        if percent_change >= self.VGPR_THRESHOLD:
            return ResponseType.VGPR, f"SPEP {percent_change:.1f}% 감소 from baseline{lcd_warning}"
        elif percent_change >= self.PR_THRESHOLD:
            return ResponseType.PR, f"SPEP {percent_change:.1f}% 감소 from baseline{lcd_warning}"
        elif percent_change >= self.MR_THRESHOLD:
            return ResponseType.MR, f"SPEP {percent_change:.1f}% 감소 from baseline{lcd_warning}"

        # Check if only percent increase condition is met (need to check other symptoms)
        if has_percent_increase and not has_absolute_increase:
            return ResponseType.SD, f"SPEP {percent_increase_from_nadir:.1f}% 증가 (다른 증상 확인 필요!){lcd_warning}"

        return ResponseType.SD, lcd_warning.strip() if lcd_warning else ""

    def _is_better_response(self, response1: ResponseType, response2: ResponseType) -> bool:
        """Check if response1 is better than response2."""
        order = [
            ResponseType.SD,
            ResponseType.MR,
            ResponseType.PR,
            ResponseType.VGPR,
            ResponseType.CR
        ]
        try:
            return order.index(response1) > order.index(response2)
        except ValueError:
            return False

    def _evaluate_lcd_type(
        self,
        lab_data: LabData,
        classification: ClassificationResult
    ) -> list[TimePointResult]:
        """
        Evaluate LCD type patients based on involved free light chain (iFLC).

        LCD Response Criteria:
        - Progression (Type Change): SPEP >= 0.5 (possible change to IgG type)
        - CR: FLC ratio (Kappa/Lambda) in normal range (0.26~1.65)
        - VGPR: iFLC >= 90% decrease from baseline OR iFLC < 100
        - PR: iFLC >= 50% decrease from baseline
        - PD: iFLC >= 25% increase from nadir AND absolute increase >= 100 from nadir
        """
        timepoints = []

        # Determine which FLC is involved
        is_kappa = classification.patient_type == PatientType.LCD_KAPPA
        baseline_flc = classification.baseline_kappa if is_kappa else classification.baseline_lambda

        if baseline_flc is None or baseline_flc == 0:
            for i in range(len(lab_data)):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=lab_data.spep[i],
                    kappa=lab_data.kappa[i],
                    lambda_=lab_data.lambda_[i],
                    upep=lab_data.upep[i] if lab_data.upep else None,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=None,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=None,
                    notes="Invalid baseline FLC value"
                ))
            return timepoints

        nadir = baseline_flc
        previous_responses: list[ResponseType] = []
        confirmed_response: Optional[ResponseType] = None
        combined_indices: set[int] = set()  # Track indices used in combinations

        for i in range(len(lab_data)):
            raw_spep = lab_data.spep[i]
            raw_kappa = lab_data.kappa[i]
            raw_lambda = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Check if this index was already combined into a later evaluation
            if i in combined_indices:
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=None,
                    confirmed_response=confirmed_response,
                    notes="→ 다음 행에서 결합 평가됨"
                ))
                continue

            # Try to get combined values within 3-day window
            spep, kappa, lambda_, used_indices, combination_note = self._get_combined_values(
                lab_data, i, combined_indices
            )

            # Mark used indices (except current) as combined and update their timepoint entries
            for idx in used_indices:
                if idx != i:
                    combined_indices.add(idx)
                    # Update the already-added timepoint to show it was combined
                    if idx < len(timepoints):
                        old_tp = timepoints[idx]
                        timepoints[idx] = TimePointResult(
                            date=old_tp.date,
                            spep=old_tp.spep,
                            kappa=old_tp.kappa,
                            lambda_=old_tp.lambda_,
                            upep=old_tp.upep,
                            percent_change_from_baseline=None,
                            change_from_nadir=None,
                            nadir_value=old_tp.nadir_value,
                            current_response=None,
                            confirmed_response=old_tp.confirmed_response,
                            notes="→ 다음 행에서 결합 평가됨"
                        )

            # Get current involved FLC value
            current_flc = kappa if is_kappa else lambda_

            if not self._is_value_valid(current_flc):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response,
                    notes="Missing FLC value"
                ))
                continue

            # Calculate percent change from baseline (positive = decrease)
            percent_change = ((baseline_flc - current_flc) / baseline_flc) * 100

            # Calculate change from nadir
            change_from_nadir = current_flc - nadir
            percent_increase_from_nadir = ((current_flc - nadir) / nadir * 100) if nadir > 0 else 0

            # Determine current response using new LCD criteria
            current_response, response_notes = self._determine_lcd_response(
                current_flc, kappa, lambda_, percent_change,
                change_from_nadir, percent_increase_from_nadir, baseline_flc, spep
            )

            # Add combination note if values were combined
            notes = response_notes
            if combination_note:
                notes = f"{combination_note} {notes}".strip()

            # Update nadir
            if current_flc < nadir:
                nadir = current_flc

            # Determine confirmed response - update when 2 consecutive same responses occur
            if previous_responses and previous_responses[-1] == current_response:
                # 2 consecutive same responses -> confirm this response
                confirmed_response = current_response
                notes = f"{current_response.value} confirmed"
                if combination_note:
                    notes = f"{combination_note} {notes}"

            timepoints.append(TimePointResult(
                date=lab_data.dates[i],
                spep=spep,
                kappa=kappa,
                lambda_=lambda_,
                upep=upep,
                percent_change_from_baseline=percent_change,
                change_from_nadir=change_from_nadir,
                nadir_value=nadir,
                current_response=current_response,
                confirmed_response=confirmed_response,
                notes=notes
            ))

            previous_responses.append(current_response)

        return timepoints

    def _determine_lcd_response(
        self,
        current_flc: float,
        kappa: Optional[float],
        lambda_: Optional[float],
        percent_change: float,
        change_from_nadir: float,
        percent_increase_from_nadir: float,
        baseline_flc: float,
        spep: Optional[float] = None
    ) -> tuple[ResponseType, str]:
        """
        Determine response type for LCD patients based on FLC value.

        LCD Response Criteria:
        - Progression (Type Change): SPEP >= 0.5 (possible change to IgG type)
        - CR: FLC ratio (Kappa/Lambda) in normal range (0.26~1.65)
        - VGPR: iFLC >= 90% decrease from baseline OR iFLC < 100
        - PR: iFLC >= 50% decrease from baseline
        - PD: iFLC >= 25% increase from nadir AND absolute increase >= 100 from nadir
        - Note: If only 25% increase is met, returns SD with "다른 증상 확인 필요!"

        Returns:
            Tuple of (ResponseType, notes string)
        """
        # Check for type change first: SPEP >= 0.5 indicates possible change to IgG type
        if spep is not None and spep >= self.SPEP_THRESHOLD:
            return ResponseType.PROGRESSION_TYPE_CHANGE, f"SPEP ({spep:.2f}) >= 0.5, possible type change to IgG"

        # Check for CR: FLC ratio in normal range (0.26~1.65)
        if self._check_flc_ratio_normal(kappa, lambda_):
            ratio = kappa / lambda_ if lambda_ and lambda_ != 0 else 0
            return ResponseType.CR, f"FLC ratio ({ratio:.2f}) normalized"

        # Check for Progression (PD):
        # iFLC >= 25% increase from nadir AND absolute increase >= 100 from nadir
        has_percent_increase = percent_increase_from_nadir >= self.LCD_PD_PERCENT_THRESHOLD
        has_absolute_increase = change_from_nadir >= self.LCD_PD_ABSOLUTE_THRESHOLD

        if has_percent_increase and has_absolute_increase:
            return ResponseType.PROGRESSION, f"iFLC increased {percent_increase_from_nadir:.1f}% from nadir (절대 증가 {change_from_nadir:.1f})"

        # Check for VGPR: iFLC >= 90% decrease from baseline OR iFLC < 100
        if percent_change >= self.LCD_VGPR_THRESHOLD:
            return ResponseType.VGPR, f"iFLC decreased {percent_change:.1f}% from baseline"
        if current_flc < self.LCD_VGPR_ABSOLUTE_THRESHOLD:
            return ResponseType.VGPR, f"iFLC ({current_flc:.1f}) < 100"

        # Check for PR: iFLC >= 50% decrease from baseline
        if percent_change >= self.LCD_PR_THRESHOLD:
            return ResponseType.PR, f"iFLC decreased {percent_change:.1f}% from baseline"

        # Check if only 25% increase condition is met (need to check other symptoms)
        if has_percent_increase and not has_absolute_increase:
            return ResponseType.SD, f"iFLC {percent_increase_from_nadir:.1f}% 증가 (다른 증상 확인 필요!)"

        # Otherwise, stable disease
        return ResponseType.SD, ""

    def _evaluate_unclassified(self, lab_data: LabData) -> list[TimePointResult]:
        """Evaluate unclassified patients - limited evaluation possible."""
        timepoints = []

        for i in range(len(lab_data)):
            timepoints.append(TimePointResult(
                date=lab_data.dates[i],
                spep=lab_data.spep[i],
                kappa=lab_data.kappa[i],
                lambda_=lab_data.lambda_[i],
                upep=lab_data.upep[i] if lab_data.upep else None,
                percent_change_from_baseline=None,
                change_from_nadir=None,
                nadir_value=None,
                current_response=ResponseType.NOT_EVALUABLE,
                confirmed_response=None,
                notes="Unclassified patient type - evaluation not available"
            ))

        return timepoints
