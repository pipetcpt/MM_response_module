"""
Response evaluator for multiple myeloma treatment.
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import math

from .classifier import PatientType, ClassificationResult
from .parser import LabData


class ResponseType(Enum):
    """Treatment response categories for multiple myeloma."""
    CR = "CR"  # Complete Response
    VGPR = "VGPR"  # Very Good Partial Response
    PR = "PR"  # Partial Response
    MR = "MR"  # Minor Response
    SD = "SD"  # Stable Disease
    PROGRESSION = "Progression"
    PROGRESSION_TYPE_CHANGE = "Progression (Type 변경 가능!)"  # Progression with possible type change
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

    For IgG type patients (SPEP >= 0.5):
    - Response is based on M-protein (SPEP) reduction from baseline
    - MR: >= 15% reduction
    - PR: >= 45% reduction
    - VGPR: >= 85% reduction
    - CR: SPEP = 0
    - Progression: increase > 0.45 g/dL from nadir
    - Progression (Type Change): After CR, if |Kappa-Lambda| > 100

    For LCD type patients (SPEP < 0.5, |Kappa-Lambda| > 100):
    - Progression (Type Change): SPEP >= 0.5 (possible change to IgG type)
    - CR: FLC ratio (Kappa/Lambda) in normal range (0.26~1.65)
    - VGPR: iFLC >= 90% decrease from baseline OR iFLC < 100
    - PR: iFLC >= 50% decrease from baseline
    - PD: iFLC >= 25% increase from nadir AND absolute increase >= 100 from nadir
    - Note: If only 25% increase → SD with "다른 증상 확인 필요!"

    Confirmation requires 2 consecutive identical responses.
    """

    # IgG Response thresholds (percentage reduction from baseline)
    MR_THRESHOLD = 15.0
    PR_THRESHOLD = 45.0
    VGPR_THRESHOLD = 85.0

    # IgG Progression threshold (absolute increase from nadir in g/dL)
    PROGRESSION_THRESHOLD = 0.45

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
        progression_confirmed = False
        cr_achieved = False  # Track if CR was ever achieved

        for i in range(len(lab_data)):
            spep = lab_data.spep[i]
            kappa = lab_data.kappa[i]
            lambda_ = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Skip if SPEP is missing
            if spep is None or (isinstance(spep, float) and math.isnan(spep)):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=spep,
                    kappa=kappa,
                    lambda_=lambda_,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response if not progression_confirmed else None,
                    notes="Missing SPEP value"
                ))
                continue

            # Calculate percent change from baseline
            percent_change = ((baseline - spep) / baseline) * 100

            # Calculate change from nadir
            change_from_nadir = spep - nadir

            # Determine current response
            current_response = self._determine_igg_response(
                spep, percent_change, change_from_nadir
            )

            # Track if CR was achieved
            if current_response == ResponseType.CR:
                cr_achieved = True

            # Check for type change possibility after CR
            # If CR was achieved but now |Kappa - Lambda| > 100, it could be LCD progression
            notes = ""
            if cr_achieved and change_from_nadir <= self.PROGRESSION_THRESHOLD:
                if self._check_type_change_possible(kappa, lambda_):
                    current_response = ResponseType.PROGRESSION_TYPE_CHANGE
                    notes = "FLC difference > 100, possible LCD type progression"

            # Update nadir if current value is lower
            if spep < nadir:
                nadir = spep

            # Determine confirmed response
            if not progression_confirmed:
                if current_response in (ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE):
                    # Check if progression is confirmed (2 consecutive)
                    if previous_responses and previous_responses[-1] in (ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE):
                        confirmed_response = current_response
                        progression_confirmed = True
                        notes = "Progression confirmed" if current_response == ResponseType.PROGRESSION else "Progression confirmed (Type 변경 가능!)"
                elif current_response in (ResponseType.CR, ResponseType.VGPR, ResponseType.PR, ResponseType.MR):
                    # Check if response is confirmed (2 consecutive)
                    if previous_responses and previous_responses[-1] == current_response:
                        confirmed_response = current_response
                        notes = f"{current_response.value} confirmed"
                    elif (previous_responses and
                          confirmed_response is not None and
                          current_response != confirmed_response):
                        # Check if new response should override confirmed
                        if self._is_better_response(current_response, confirmed_response):
                            if previous_responses[-1] == current_response:
                                confirmed_response = current_response
                                notes = f"{current_response.value} confirmed (upgraded)"

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
                confirmed_response=confirmed_response if not (progression_confirmed and current_response not in (ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE)) else (confirmed_response if progression_confirmed else None),
                notes=notes
            ))

            previous_responses.append(current_response)

        return timepoints

    def _determine_igg_response(
        self,
        spep: float,
        percent_change: float,
        change_from_nadir: float
    ) -> ResponseType:
        """Determine response type for IgG patients based on SPEP value."""
        # Check for CR first
        if spep == 0:
            return ResponseType.CR

        # Check for Progression (increase > 0.45 from nadir)
        if change_from_nadir > self.PROGRESSION_THRESHOLD:
            return ResponseType.PROGRESSION

        # Check response depth based on percent change from baseline
        if percent_change >= self.VGPR_THRESHOLD:
            return ResponseType.VGPR
        elif percent_change >= self.PR_THRESHOLD:
            return ResponseType.PR
        elif percent_change >= self.MR_THRESHOLD:
            return ResponseType.MR
        else:
            return ResponseType.SD

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
        progression_confirmed = False

        for i in range(len(lab_data)):
            spep = lab_data.spep[i]
            kappa = lab_data.kappa[i]
            lambda_ = lab_data.lambda_[i]
            upep = lab_data.upep[i] if lab_data.upep else None

            # Get current involved FLC value
            current_flc = kappa if is_kappa else lambda_

            if current_flc is None or (isinstance(current_flc, float) and math.isnan(current_flc)):
                timepoints.append(TimePointResult(
                    date=lab_data.dates[i],
                    spep=spep,
                    kappa=kappa,
                    lambda_=lambda_,
                    upep=upep,
                    percent_change_from_baseline=None,
                    change_from_nadir=None,
                    nadir_value=nadir,
                    current_response=ResponseType.NOT_EVALUABLE,
                    confirmed_response=confirmed_response if not progression_confirmed else None,
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

            # Update nadir
            if current_flc < nadir:
                nadir = current_flc

            # Confirmation logic (same as IgG)
            notes = response_notes
            if not progression_confirmed:
                if current_response in (ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE):
                    if previous_responses and previous_responses[-1] in (ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE):
                        confirmed_response = current_response
                        progression_confirmed = True
                        notes = "Progression confirmed" if current_response == ResponseType.PROGRESSION else "Progression confirmed (Type 변경 가능!)"
                elif current_response in (ResponseType.CR, ResponseType.VGPR, ResponseType.PR, ResponseType.MR):
                    if previous_responses and previous_responses[-1] == current_response:
                        confirmed_response = current_response
                        notes = f"{current_response.value} confirmed"

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
