"""
Patient type classifier for multiple myeloma.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass


class PatientType(Enum):
    """Multiple myeloma patient types based on M-protein and FLC."""
    IGG_KAPPA = "IgG_Kappa"
    IGG_LAMBDA = "IgG_Lambda"
    LCD_KAPPA = "LCD_Kappa"  # Light Chain Disease
    LCD_LAMBDA = "LCD_Lambda"
    UNCLASSIFIED = "Unclassified"

    def is_igg_type(self) -> bool:
        """Check if this is an IgG type (measurable M-protein)."""
        return self in (PatientType.IGG_KAPPA, PatientType.IGG_LAMBDA)

    def is_lcd_type(self) -> bool:
        """Check if this is a Light Chain Disease type."""
        return self in (PatientType.LCD_KAPPA, PatientType.LCD_LAMBDA)

    def is_kappa(self) -> bool:
        """Check if this is a Kappa-predominant type."""
        return self in (PatientType.IGG_KAPPA, PatientType.LCD_KAPPA)

    def is_lambda(self) -> bool:
        """Check if this is a Lambda-predominant type."""
        return self in (PatientType.IGG_LAMBDA, PatientType.LCD_LAMBDA)


@dataclass
class ClassificationResult:
    """Result of patient classification."""
    patient_type: PatientType
    baseline_spep: Optional[float]
    baseline_kappa: Optional[float]
    baseline_lambda: Optional[float]
    classification_reason: str


class PatientClassifier:
    """
    Classifier for multiple myeloma patient types.

    Classification Rules:
    1. If SPEP (M-protein) >= 0.5 g/dL:
       - Kappa > Lambda → IgG_Kappa
       - Lambda >= Kappa → IgG_Lambda
    2. If SPEP < 0.5 g/dL and |Kappa - Lambda| > 100:
       - Kappa > Lambda → LCD_Kappa
       - Lambda > Kappa → LCD_Lambda
    3. Otherwise → Unclassified
    """

    SPEP_THRESHOLD = 0.5  # g/dL
    FLC_DIFFERENCE_THRESHOLD = 100  # mg/L

    def classify(
        self,
        spep: Optional[float],
        kappa: Optional[float],
        lambda_: Optional[float]
    ) -> ClassificationResult:
        """
        Classify patient type based on baseline values.

        Args:
            spep: Serum M-protein value (g/dL)
            kappa: Serum free Kappa light chain (mg/L)
            lambda_: Serum free Lambda light chain (mg/L)

        Returns:
            ClassificationResult with patient type and reasoning
        """
        # Validate required values
        if spep is None or kappa is None or lambda_ is None:
            return ClassificationResult(
                patient_type=PatientType.UNCLASSIFIED,
                baseline_spep=spep,
                baseline_kappa=kappa,
                baseline_lambda=lambda_,
                classification_reason="Missing required baseline values"
            )

        # IgG type classification (SPEP >= 0.5)
        if spep >= self.SPEP_THRESHOLD:
            if kappa > lambda_:
                return ClassificationResult(
                    patient_type=PatientType.IGG_KAPPA,
                    baseline_spep=spep,
                    baseline_kappa=kappa,
                    baseline_lambda=lambda_,
                    classification_reason=f"SPEP ({spep:.2f}) >= {self.SPEP_THRESHOLD}, Kappa ({kappa:.2f}) > Lambda ({lambda_:.2f})"
                )
            else:
                return ClassificationResult(
                    patient_type=PatientType.IGG_LAMBDA,
                    baseline_spep=spep,
                    baseline_kappa=kappa,
                    baseline_lambda=lambda_,
                    classification_reason=f"SPEP ({spep:.2f}) >= {self.SPEP_THRESHOLD}, Lambda ({lambda_:.2f}) >= Kappa ({kappa:.2f})"
                )

        # LCD type classification (SPEP < 0.5 and FLC difference > 100)
        flc_diff = abs(kappa - lambda_)
        if flc_diff > self.FLC_DIFFERENCE_THRESHOLD:
            if kappa > lambda_:
                return ClassificationResult(
                    patient_type=PatientType.LCD_KAPPA,
                    baseline_spep=spep,
                    baseline_kappa=kappa,
                    baseline_lambda=lambda_,
                    classification_reason=f"SPEP ({spep:.2f}) < {self.SPEP_THRESHOLD}, |Kappa - Lambda| ({flc_diff:.2f}) > {self.FLC_DIFFERENCE_THRESHOLD}, Kappa dominant"
                )
            else:
                return ClassificationResult(
                    patient_type=PatientType.LCD_LAMBDA,
                    baseline_spep=spep,
                    baseline_kappa=kappa,
                    baseline_lambda=lambda_,
                    classification_reason=f"SPEP ({spep:.2f}) < {self.SPEP_THRESHOLD}, |Kappa - Lambda| ({flc_diff:.2f}) > {self.FLC_DIFFERENCE_THRESHOLD}, Lambda dominant"
                )

        # Unclassified
        return ClassificationResult(
            patient_type=PatientType.UNCLASSIFIED,
            baseline_spep=spep,
            baseline_kappa=kappa,
            baseline_lambda=lambda_,
            classification_reason=f"SPEP ({spep:.2f}) < {self.SPEP_THRESHOLD}, |Kappa - Lambda| ({flc_diff:.2f}) < {self.FLC_DIFFERENCE_THRESHOLD}"
        )
