"""
Main application for Multiple Myeloma Response Evaluation.
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from .parser import ExcelParser, LabData
from .classifier import PatientClassifier, ClassificationResult
from .evaluator import ResponseEvaluator, EvaluationResult, ResponseType


def evaluate_file(file_path: str, verbose: bool = False) -> EvaluationResult:
    """
    Evaluate a single patient file.

    Args:
        file_path: Path to the Excel file
        verbose: Whether to print detailed output

    Returns:
        EvaluationResult with all evaluation data
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(file_path)}")
        print('='*60)

    # Parse Excel file
    parser = ExcelParser(file_path)
    lab_data = parser.parse()

    if verbose:
        print(f"Found {len(lab_data)} timepoints")

    # Classify patient type
    classifier = PatientClassifier()
    baseline_spep, baseline_kappa, baseline_lambda = lab_data.get_baseline_values()
    classification = classifier.classify(baseline_spep, baseline_kappa, baseline_lambda)

    if verbose:
        print(f"\nPatient Classification:")
        print(f"  Type: {classification.patient_type.value}")
        print(f"  Reason: {classification.classification_reason}")
        print(f"  Baseline SPEP: {baseline_spep}")
        print(f"  Baseline Kappa: {baseline_kappa}")
        print(f"  Baseline Lambda: {baseline_lambda}")

    # Evaluate response
    evaluator = ResponseEvaluator()
    result = evaluator.evaluate(lab_data, classification)

    return result


def print_results(result: EvaluationResult) -> None:
    """Print evaluation results in a formatted table."""
    print(f"\n{'='*100}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*100}")
    print(f"Patient Type: {result.patient_type.value}")
    print(f"Classification: {result.classification_reason}")
    print(f"\nBaseline Values:")
    print(f"  SPEP: {result.baseline_spep}")
    print(f"  Kappa: {result.baseline_kappa}")
    print(f"  Lambda: {result.baseline_lambda}")

    print(f"\n{'-'*100}")
    print(f"{'Date':<20} {'SPEP':>8} {'Kappa':>10} {'Lambda':>10} {'%Change':>10} {'Nadir':>8} {'Current':>12} {'Confirmed':>12}")
    print(f"{'-'*100}")

    for tp in result.timepoints:
        date_str = tp.date.strftime("%Y-%m-%d") if tp.date else "N/A"
        spep_str = f"{tp.spep:.2f}" if tp.spep is not None else "N/A"
        kappa_str = f"{tp.kappa:.2f}" if tp.kappa is not None else "N/A"
        lambda_str = f"{tp.lambda_:.2f}" if tp.lambda_ is not None else "N/A"
        pct_str = f"{tp.percent_change_from_baseline:.1f}%" if tp.percent_change_from_baseline is not None else "N/A"
        nadir_str = f"{tp.nadir_value:.2f}" if tp.nadir_value is not None else "N/A"
        current_str = tp.current_response.value if tp.current_response else "N/A"
        confirmed_str = tp.confirmed_response.value if tp.confirmed_response else ""

        print(f"{date_str:<20} {spep_str:>8} {kappa_str:>10} {lambda_str:>10} {pct_str:>10} {nadir_str:>8} {current_str:>12} {confirmed_str:>12}")

    print(f"{'-'*100}")


def results_to_dataframe(result: EvaluationResult) -> pd.DataFrame:
    """Convert evaluation results to a pandas DataFrame."""
    data = []
    for tp in result.timepoints:
        data.append({
            "Date": tp.date.strftime("%Y-%m-%d %H:%M") if tp.date else None,
            "SPEP": tp.spep,
            "Kappa": tp.kappa,
            "Lambda": tp.lambda_,
            "UPEP": tp.upep,
            "%Change from Baseline": round(tp.percent_change_from_baseline, 2) if tp.percent_change_from_baseline is not None else None,
            "Change from Nadir": round(tp.change_from_nadir, 4) if tp.change_from_nadir is not None else None,
            "Nadir": tp.nadir_value,
            "Current Response": tp.current_response.value if tp.current_response else None,
            "Confirmed Response": tp.confirmed_response.value if tp.confirmed_response else None,
            "Notes": tp.notes
        })

    df = pd.DataFrame(data)
    return df


def export_results(result: EvaluationResult, output_path: str) -> None:
    """Export evaluation results to an Excel file."""
    df = results_to_dataframe(result)

    # Add summary sheet
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Property": ["Patient Type", "Classification Reason", "Baseline SPEP", "Baseline Kappa", "Baseline Lambda"],
            "Value": [
                result.patient_type.value,
                result.classification_reason,
                result.baseline_spep,
                result.baseline_kappa,
                result.baseline_lambda
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Detailed results sheet
        df.to_excel(writer, sheet_name="Response Evaluation", index=False)

    print(f"Results exported to: {output_path}")


def process_folder(folder_path: str, output_folder: str = None, verbose: bool = False) -> dict:
    """
    Process all Excel files in a folder.

    Args:
        folder_path: Path to folder containing Excel files
        output_folder: Path to output folder (optional)
        verbose: Whether to print detailed output

    Returns:
        Dictionary mapping file names to their evaluation results
    """
    results = {}
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    excel_files = list(folder.glob("*.xlsx"))
    print(f"Found {len(excel_files)} Excel files in {folder_path}")

    for file_path in sorted(excel_files):
        try:
            result = evaluate_file(str(file_path), verbose=verbose)
            results[file_path.name] = result

            if verbose:
                print_results(result)

            if output_folder:
                output_path = Path(output_folder) / f"{file_path.stem}_evaluated.xlsx"
                export_results(result, str(output_path))

        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    return results


def create_summary_report(results: dict, output_path: str) -> None:
    """Create a summary report of all processed files."""
    summary_data = []

    for filename, result in results.items():
        # Find the best confirmed response and latest status
        best_response = None
        final_response = None
        progression_date = None

        for tp in result.timepoints:
            if tp.confirmed_response:
                final_response = tp.confirmed_response
                if tp.confirmed_response == ResponseType.PROGRESSION:
                    progression_date = tp.date

                # Track best response (excluding progression)
                if tp.confirmed_response != ResponseType.PROGRESSION:
                    if best_response is None:
                        best_response = tp.confirmed_response
                    else:
                        # Compare responses
                        order = [ResponseType.SD, ResponseType.MR, ResponseType.PR, ResponseType.VGPR, ResponseType.CR]
                        try:
                            if order.index(tp.confirmed_response) > order.index(best_response):
                                best_response = tp.confirmed_response
                        except ValueError:
                            pass

        summary_data.append({
            "File": filename,
            "Patient Type": result.patient_type.value,
            "Baseline SPEP": result.baseline_spep,
            "Baseline Kappa": result.baseline_kappa,
            "Baseline Lambda": result.baseline_lambda,
            "Best Response": best_response.value if best_response else "N/A",
            "Final Status": final_response.value if final_response else "N/A",
            "Progression Date": progression_date.strftime("%Y-%m-%d") if progression_date else "",
            "Total Timepoints": len(result.timepoints)
        })

    df = pd.DataFrame(summary_data)
    df.to_excel(output_path, index=False)
    print(f"Summary report saved to: {output_path}")


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Multiple Myeloma Response Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single file
  python -m myeloma_response.main -f patient_data.xlsx

  # Evaluate all files in a folder
  python -m myeloma_response.main -d Data/Training\ folder/

  # Evaluate and export results
  python -m myeloma_response.main -f patient_data.xlsx -o results.xlsx

  # Batch process with summary report
  python -m myeloma_response.main -d Data/Training\ folder/ --summary summary.xlsx
        """
    )

    parser.add_argument(
        "-f", "--file",
        type=str,
        help="Path to a single Excel file to evaluate"
    )

    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="Path to a directory containing Excel files"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Output path for results (Excel file)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch processing"
    )

    parser.add_argument(
        "--summary",
        type=str,
        help="Path for summary report (Excel file)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    if not args.file and not args.directory:
        parser.print_help()
        sys.exit(1)

    try:
        if args.file:
            # Single file processing
            result = evaluate_file(args.file, verbose=args.verbose)
            print_results(result)

            if args.output:
                export_results(result, args.output)

        elif args.directory:
            # Batch processing
            results = process_folder(
                args.directory,
                output_folder=args.output_dir,
                verbose=args.verbose
            )

            if args.summary:
                create_summary_report(results, args.summary)

            print(f"\nProcessed {len(results)} files successfully.")

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
