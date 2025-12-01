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
    segment_id: int = 0  # Which treatment segment this belongs to
    is_new_baseline: bool = False  # Whether this is a new baseline for a segment


@dataclass
class TreatmentSegment:
    """Represents a treatment period with its own baseline."""
    segment_id: int
    start_date: datetime
    patient_type: PatientType
    baseline_spep: Optional[float]
    baseline_kappa: Optional[float]
    baseline_lambda: Optional[float]
    is_type_override: bool = False  # Whether type was manually overridden


@dataclass
class EvaluationResult:
    """Complete evaluation result for a patient."""
    patient_type: PatientType
    classification_reason: str
    baseline_spep: Optional[float]
    baseline_kappa: Optional[float]
    baseline_lambda: Optional[float]
    timepoints: list[TimePointResult] = field(default_factory=list)
    segments: list[TreatmentSegment] = field(default_factory=list)  # Treatment segments


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
        classification: ClassificationResult,
        treatment_changes: Optional[list[datetime]] = None,
        type_overrides: Optional[dict[datetime, PatientType]] = None
    ) -> EvaluationResult:
        """
        Evaluate treatment response for a patient.

        Args:
            lab_data: Parsed laboratory data
            classification: Patient classification result
            treatment_changes: List of dates where new treatment started (re-baseline)
            type_overrides: Dict mapping dates to patient types for manual type changes

        Returns:
            EvaluationResult with all timepoint evaluations
        """
        treatment_changes = treatment_changes or []
        type_overrides = type_overrides or {}

        # Sort treatment changes
        treatment_changes = sorted(treatment_changes)

        # Build segments based on treatment changes and type overrides
        segments = self._build_segments(
            lab_data, classification, treatment_changes, type_overrides
        )

        result = EvaluationResult(
            patient_type=classification.patient_type,
            classification_reason=classification.classification_reason,
            baseline_spep=classification.baseline_spep,
            baseline_kappa=classification.baseline_kappa,
            baseline_lambda=classification.baseline_lambda,
            segments=segments
        )

        # Evaluate each segment
        all_timepoints = []
        for i, segment in enumerate(segments):
            # Get next segment's start date to limit this segment's date range
            next_segment_start = segments[i + 1].start_date if i + 1 < len(segments) else None
            segment_timepoints = self._evaluate_segment(lab_data, segment, next_segment_start)
            all_timepoints.extend(segment_timepoints)

        result.timepoints = all_timepoints
        return result

    def _build_segments(
        self,
        lab_data: LabData,
        classification: ClassificationResult,
        treatment_changes: list[datetime],
        type_overrides: dict[datetime, PatientType]
    ) -> list[TreatmentSegment]:
        """Build treatment segments based on treatment changes and type overrides."""
        segments = []

        # Collect all segment boundary dates
        boundary_dates = set()
        boundary_dates.add(lab_data.dates[0])  # Initial date

        for tc_date in treatment_changes:
            boundary_dates.add(tc_date)

        for to_date in type_overrides.keys():
            boundary_dates.add(to_date)

        boundary_dates = sorted(boundary_dates)

        # Build segments
        for seg_idx, start_date in enumerate(boundary_dates):
            # Find the data index for this start date
            start_idx = None
            for i, d in enumerate(lab_data.dates):
                if d >= start_date:
                    start_idx = i
                    break

            if start_idx is None:
                continue

            # Determine patient type for this segment
            if start_date in type_overrides:
                patient_type = type_overrides[start_date]
                is_type_override = True
            elif seg_idx == 0:
                patient_type = classification.patient_type
                is_type_override = False
            else:
                # Inherit from previous segment or use classification
                if segments:
                    patient_type = segments[-1].patient_type
                else:
                    patient_type = classification.patient_type
                is_type_override = False

            # Get baseline values at segment start
            baseline_spep = lab_data.spep[start_idx]
            baseline_kappa = lab_data.kappa[start_idx]
            baseline_lambda = lab_data.lambda_[start_idx]

            # Try to get valid baseline values from nearby dates (within 3 days)
            if not self._is_value_valid(baseline_spep) or not self._is_value_valid(baseline_kappa) or not self._is_value_valid(baseline_lambda):
                for j in range(start_idx, min(start_idx + 5, len(lab_data.dates))):
                    if not self._is_value_valid(baseline_spep) and self._is_value_valid(lab_data.spep[j]):
                        baseline_spep = lab_data.spep[j]
                    if not self._is_value_valid(baseline_kappa) and self._is_value_valid(lab_data.kappa[j]):
                        baseline_kappa = lab_data.kappa[j]
                    if not self._is_value_valid(baseline_lambda) and self._is_value_valid(lab_data.lambda_[j]):
                        baseline_lambda = lab_data.lambda_[j]

            segment = TreatmentSegment(
                segment_id=seg_idx,
                start_date=start_date,
                patient_type=patient_type,
                baseline_spep=baseline_spep,
                baseline_kappa=baseline_kappa,
                baseline_lambda=baseline_lambda,
                is_type_override=is_type_override
            )
            segments.append(segment)

        return segments

    def _get_segment_for_date(self, date: datetime, segments: list[TreatmentSegment]) -> TreatmentSegment:
        """Get the appropriate segment for a given date."""
        result_segment = segments[0]
        for segment in segments:
            if date >= segment.start_date:
                result_segment = segment
        return result_segment

    def _evaluate_segment(
        self,
        lab_data: LabData,
        segment: TreatmentSegment,
        next_segment_start: datetime = None
    ) -> list[TimePointResult]:
        """Evaluate a single treatment segment."""
        # Create a classification-like object for the segment
        class SegmentClassification:
            def __init__(self, seg: TreatmentSegment):
                self.patient_type = seg.patient_type
                self.baseline_spep = seg.baseline_spep
                self.baseline_kappa = seg.baseline_kappa
                self.baseline_lambda = seg.baseline_lambda

        seg_class = SegmentClassification(segment)

        # Find indices for this segment
        start_idx = None
        end_idx = len(lab_data.dates)

        for i, d in enumerate(lab_data.dates):
            if d >= segment.start_date:
                if start_idx is None:
                    start_idx = i
            # Find end index (before next segment starts)
            if next_segment_start and d >= next_segment_start:
                end_idx = i
                break

        if start_idx is None:
            return []

        # Evaluate with the segment's date range
        if segment.patient_type.is_igg_type():
            timepoints = self._evaluate_igg_type_segment(lab_data, seg_class, segment, end_idx)
        elif segment.patient_type.is_lcd_type():
            timepoints = self._evaluate_lcd_type_segment(lab_data, seg_class, segment, end_idx)
        else:
            timepoints = self._evaluate_unclassified_segment(lab_data, segment, end_idx)

        return timepoints

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

    def _evaluate_igg_type_segment(
        self,
        lab_data: LabData,
        classification,
        segment: TreatmentSegment,
        end_idx: int = None
    ) -> list[TimePointResult]:
        """Evaluate IgG type patients for a specific treatment segment."""
        timepoints = []
        baseline = classification.baseline_spep

        # Find segment date range
        start_idx = None
        for i, d in enumerate(lab_data.dates):
            if d >= segment.start_date:
                start_idx = i
                break

        if start_idx is None:
            return []

        if end_idx is None:
            end_idx = len(lab_data)

        if baseline is None or baseline == 0:
            # Cannot evaluate without valid baseline
            for i in range(start_idx, end_idx):
                is_first = (i == start_idx)
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
                    notes="Invalid baseline SPEP value",
                    segment_id=segment.segment_id,
                    is_new_baseline=is_first and segment.segment_id > 0
                ))
            return timepoints

        # Track nadir (minimum value) for progression detection
        nadir = baseline
        previous_responses: list[ResponseType] = []
        confirmed_response: Optional[ResponseType] = None
        combined_indices: set[int] = set()

        for i in range(start_idx, end_idx):
            current_date = lab_data.dates[i]
            is_first = (i == start_idx)

            raw_spep = lab_data.spep[i]
            raw_kappa = lab_data.kappa[i]
            raw_lambda = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Check if this index was already combined
            if i in combined_indices:
                timepoints.append(TimePointResult(
                    date=current_date,
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=None,
                    confirmed_response=confirmed_response,
                    notes="→ 다음 행에서 결합 평가됨",
                    segment_id=segment.segment_id,
                    is_new_baseline=False
                ))
                continue

            # Try to get combined values within 3-day window
            spep, kappa, lambda_, used_indices, combination_note = self._get_combined_values(
                lab_data, i, combined_indices
            )

            # Mark used indices as combined
            for idx in used_indices:
                if idx != i:
                    combined_indices.add(idx)
                    if idx < len(timepoints):
                        old_tp = timepoints[idx - start_idx] if idx >= start_idx else None
                        if old_tp:
                            tp_idx = idx - start_idx
                            timepoints[tp_idx] = TimePointResult(
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
                                notes="→ 다음 행에서 결합 평가됨",
                                segment_id=segment.segment_id,
                                is_new_baseline=old_tp.is_new_baseline
                            )

            # Skip if SPEP is still missing
            if not self._is_value_valid(spep):
                timepoints.append(TimePointResult(
                    date=current_date,
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response,
                    notes="Missing SPEP value",
                    segment_id=segment.segment_id,
                    is_new_baseline=is_first and segment.segment_id > 0
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

            # Add combination note and segment note if applicable
            notes = response_notes
            if combination_note:
                notes = f"{combination_note} {notes}".strip()
            if is_first and segment.segment_id > 0:
                segment_note = f"[Segment {segment.segment_id + 1} 시작 - 새 Baseline]"
                notes = f"{segment_note} {notes}".strip()

            # Update nadir if current value is lower
            if spep < nadir:
                nadir = spep

            # Determine confirmed response
            if previous_responses and previous_responses[-1] == current_response:
                confirmed_response = current_response
                lcd_warn = " (LCD Type 변경 확인!)" if "(LCD Type 변경 확인!)" in response_notes else ""
                notes = f"{current_response.value} confirmed{lcd_warn}"
                if combination_note:
                    notes = f"{combination_note} {notes}"

            timepoints.append(TimePointResult(
                date=current_date,
                spep=spep,
                kappa=kappa,
                lambda_=lambda_,
                upep=upep,
                percent_change_from_baseline=percent_change,
                change_from_nadir=change_from_nadir,
                nadir_value=nadir,
                current_response=current_response,
                confirmed_response=confirmed_response,
                notes=notes,
                segment_id=segment.segment_id,
                is_new_baseline=is_first and segment.segment_id > 0
            ))

            previous_responses.append(current_response)

        return timepoints

    def _evaluate_lcd_type_segment(
        self,
        lab_data: LabData,
        classification,
        segment: TreatmentSegment,
        end_idx: int = None
    ) -> list[TimePointResult]:
        """Evaluate LCD type patients for a specific treatment segment."""
        timepoints = []

        # Determine which FLC is involved
        is_kappa = segment.patient_type == PatientType.LCD_KAPPA
        baseline_flc = classification.baseline_kappa if is_kappa else classification.baseline_lambda

        # Find segment date range
        start_idx = None
        for i, d in enumerate(lab_data.dates):
            if d >= segment.start_date:
                start_idx = i
                break

        if start_idx is None:
            return []

        if end_idx is None:
            end_idx = len(lab_data)

        if baseline_flc is None or baseline_flc == 0:
            for i in range(start_idx, end_idx):
                is_first = (i == start_idx)
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
                    notes="Invalid baseline FLC value",
                    segment_id=segment.segment_id,
                    is_new_baseline=is_first and segment.segment_id > 0
                ))
            return timepoints

        nadir = baseline_flc
        previous_responses: list[ResponseType] = []
        confirmed_response: Optional[ResponseType] = None
        combined_indices: set[int] = set()

        for i in range(start_idx, end_idx):
            current_date = lab_data.dates[i]
            is_first = (i == start_idx)

            raw_spep = lab_data.spep[i]
            raw_kappa = lab_data.kappa[i]
            raw_lambda = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Check if this index was already combined
            if i in combined_indices:
                timepoints.append(TimePointResult(
                    date=current_date,
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=None,
                    confirmed_response=confirmed_response,
                    notes="→ 다음 행에서 결합 평가됨",
                    segment_id=segment.segment_id,
                    is_new_baseline=False
                ))
                continue

            # Try to get combined values
            spep, kappa, lambda_, used_indices, combination_note = self._get_combined_values(
                lab_data, i, combined_indices
            )

            # Mark used indices as combined
            for idx in used_indices:
                if idx != i:
                    combined_indices.add(idx)

            # Get current involved FLC value
            current_flc = kappa if is_kappa else lambda_

            if not self._is_value_valid(current_flc):
                timepoints.append(TimePointResult(
                    date=current_date,
                    spep=raw_spep,
                    kappa=raw_kappa,
                    lambda_=raw_lambda,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response,
                    notes="Missing FLC value",
                    segment_id=segment.segment_id,
                    is_new_baseline=is_first and segment.segment_id > 0
                ))
                continue

            # Calculate percent change from baseline
            percent_change = ((baseline_flc - current_flc) / baseline_flc) * 100

            # Calculate change from nadir
            change_from_nadir = current_flc - nadir
            percent_increase_from_nadir = ((current_flc - nadir) / nadir * 100) if nadir > 0 else 0

            # Determine current response
            current_response, response_notes = self._determine_lcd_response(
                current_flc, kappa, lambda_, percent_change,
                change_from_nadir, percent_increase_from_nadir, baseline_flc, spep
            )

            # Add combination note and segment note if applicable
            notes = response_notes
            if combination_note:
                notes = f"{combination_note} {notes}".strip()
            if is_first and segment.segment_id > 0:
                segment_note = f"[Segment {segment.segment_id + 1} 시작 - 새 Baseline]"
                notes = f"{segment_note} {notes}".strip()

            # Update nadir
            if current_flc < nadir:
                nadir = current_flc

            # Determine confirmed response
            if previous_responses and previous_responses[-1] == current_response:
                confirmed_response = current_response
                notes = f"{current_response.value} confirmed"
                if combination_note:
                    notes = f"{combination_note} {notes}"

            timepoints.append(TimePointResult(
                date=current_date,
                spep=spep,
                kappa=kappa,
                lambda_=lambda_,
                upep=upep,
                percent_change_from_baseline=percent_change,
                change_from_nadir=change_from_nadir,
                nadir_value=nadir,
                current_response=current_response,
                confirmed_response=confirmed_response,
                notes=notes,
                segment_id=segment.segment_id,
                is_new_baseline=is_first and segment.segment_id > 0
            ))

            previous_responses.append(current_response)

        return timepoints

    def _evaluate_unclassified_segment(
        self,
        lab_data: LabData,
        segment: TreatmentSegment,
        end_idx: int = None
    ) -> list[TimePointResult]:
        """Evaluate unclassified patients for a specific segment."""
        timepoints = []

        start_idx = None
        for i, d in enumerate(lab_data.dates):
            if d >= segment.start_date:
                start_idx = i
                break

        if start_idx is None:
            return []

        if end_idx is None:
            end_idx = len(lab_data)

        for i in range(start_idx, end_idx):
            is_first = (i == start_idx)
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
                notes="Unclassified patient type - evaluation not available",
                segment_id=segment.segment_id,
                is_new_baseline=is_first and segment.segment_id > 0
            ))

        return timepoints
