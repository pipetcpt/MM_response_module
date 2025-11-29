"""
Streamlit Web Application for Multiple Myeloma Response Evaluation

Upload an Excel file and get treatment response evaluation results.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from io import BytesIO

# Import our modules
from myeloma_response.parser import ExcelParser
from myeloma_response.classifier import PatientClassifier
from myeloma_response.evaluator import ResponseEvaluator, ResponseType


def evaluate_uploaded_file(uploaded_file):
    """Evaluate an uploaded Excel file."""
    # Save uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Parse Excel file
        parser = ExcelParser(tmp_path)
        lab_data = parser.parse()

        # Classify patient type
        classifier = PatientClassifier()
        baseline_spep, baseline_kappa, baseline_lambda = lab_data.get_baseline_values()
        classification = classifier.classify(baseline_spep, baseline_kappa, baseline_lambda)

        # Evaluate response
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(lab_data, classification)

        return result, None
    except Exception as e:
        return None, str(e)
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def create_results_dataframe(result):
    """Convert evaluation results to a pandas DataFrame."""
    data = []
    for idx, tp in enumerate(result.timepoints):
        data.append({
            "Timepoint": idx + 1,
            "Date": tp.date.strftime("%Y-%m-%d") if tp.date else None,
            "SPEP": tp.spep,
            "Kappa": tp.kappa,
            "Lambda": tp.lambda_,
            "UPEP": tp.upep,
            "%Change": round(tp.percent_change_from_baseline, 1) if tp.percent_change_from_baseline is not None else None,
            "Nadir": tp.nadir_value,
            "Current Response": tp.current_response.value if tp.current_response else None,
            "Confirmed Response": tp.confirmed_response.value if tp.confirmed_response else None,
        })
    return pd.DataFrame(data)


def get_response_summary(result):
    """Get summary of key response dates."""
    best_response = None
    best_response_date = None
    cr_date = None
    progression_date = None

    for tp in result.timepoints:
        if tp.confirmed_response:
            if tp.confirmed_response == ResponseType.CR and cr_date is None:
                cr_date = tp.date
            if tp.confirmed_response == ResponseType.PROGRESSION and progression_date is None:
                progression_date = tp.date
            if tp.confirmed_response != ResponseType.PROGRESSION:
                order = [ResponseType.SD, ResponseType.MR, ResponseType.PR, ResponseType.VGPR, ResponseType.CR]
                if best_response is None or (tp.confirmed_response in order and
                    order.index(tp.confirmed_response) > order.index(best_response)):
                    best_response = tp.confirmed_response
                    best_response_date = tp.date

    return {
        "best_response": best_response,
        "best_response_date": best_response_date,
        "cr_date": cr_date,
        "progression_date": progression_date
    }


def export_to_excel(result):
    """Export results to Excel bytes for download."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        summary_data = {
            "Property": [
                "Patient Type",
                "Baseline SPEP",
                "Baseline Kappa",
                "Baseline Lambda"
            ],
            "Value": [
                result.patient_type.value,
                result.baseline_spep,
                result.baseline_kappa,
                result.baseline_lambda
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Detailed results
        df = create_results_dataframe(result)
        df.to_excel(writer, sheet_name="Response Evaluation", index=False)

    output.seek(0)
    return output


def main():
    st.set_page_config(
        page_title="MM Response Evaluator",
        page_icon="🔬",
        layout="wide"
    )

    st.title("🔬 Multiple Myeloma Response Evaluator")
    st.markdown("""
    다발골수종 치료 반응 평가 도구입니다. Excel 파일을 업로드하면 SPEP, Kappa, Lambda 값을 분석하여
    치료 반응(CR, VGPR, PR, MR, Progression)을 평가합니다.
    """)

    # File upload
    st.sidebar.header("📁 파일 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "Excel 파일을 선택하세요",
        type=['xlsx', 'xls'],
        help="SPEP, serum Kappa, serum Lambda 데이터가 포함된 Excel 파일"
    )

    if uploaded_file is not None:
        st.sidebar.success(f"✅ 파일 업로드됨: {uploaded_file.name}")

        # Evaluate
        with st.spinner("분석 중..."):
            result, error = evaluate_uploaded_file(uploaded_file)

        if error:
            st.error(f"❌ 오류 발생: {error}")
        else:
            # Display results
            st.header("📊 분석 결과")

            # Patient classification
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("환자 분류")
                st.metric("Patient Type", result.patient_type.value)
                st.caption(result.classification_reason)

            with col2:
                st.subheader("Baseline Values")
                baseline_col1, baseline_col2, baseline_col3 = st.columns(3)
                baseline_col1.metric("SPEP", f"{result.baseline_spep:.2f}" if result.baseline_spep else "N/A")
                baseline_col2.metric("Kappa", f"{result.baseline_kappa:.2f}" if result.baseline_kappa else "N/A")
                baseline_col3.metric("Lambda", f"{result.baseline_lambda:.2f}" if result.baseline_lambda else "N/A")

            # Response summary
            st.subheader("📈 Response Summary")
            summary = get_response_summary(result)

            resp_col1, resp_col2, resp_col3 = st.columns(3)

            with resp_col1:
                best_resp_text = summary["best_response"].value if summary["best_response"] else "N/A"
                best_resp_date = summary["best_response_date"].strftime("%Y-%m-%d") if summary["best_response_date"] else ""
                st.metric("Best Response", best_resp_text, best_resp_date)

            with resp_col2:
                if summary["cr_date"]:
                    st.metric("CR Achieved", summary["cr_date"].strftime("%Y-%m-%d"))
                else:
                    st.metric("CR Achieved", "Not achieved")

            with resp_col3:
                if summary["progression_date"]:
                    st.metric("Progression", summary["progression_date"].strftime("%Y-%m-%d"), delta="⚠️", delta_color="inverse")
                else:
                    st.metric("Progression", "None")

            # Serial results table
            st.subheader("📋 Serial Response Evaluation")
            df = create_results_dataframe(result)

            # Try to style the dataframe (requires jinja2)
            try:
                def highlight_response(val):
                    if val == "CR":
                        return "background-color: #90EE90"  # Light green
                    elif val == "VGPR":
                        return "background-color: #98FB98"  # Pale green
                    elif val == "PR":
                        return "background-color: #FFFFE0"  # Light yellow
                    elif val == "MR":
                        return "background-color: #FAFAD2"  # Light goldenrod
                    elif val == "Progression":
                        return "background-color: #FFB6C1"  # Light pink
                    elif val == "Progression (Type 변경 가능!)":
                        return "background-color: #FFA500"  # Orange - type change warning
                    elif val == "NE":
                        return "background-color: #D3D3D3"  # Light gray
                    return ""

                styled_df = df.style.map(
                    highlight_response,
                    subset=["Current Response", "Confirmed Response"]
                )
                st.dataframe(styled_df, use_container_width=True, height=400)
            except (ImportError, AttributeError):
                # Fallback: show without styling if jinja2 is not installed
                st.dataframe(df, use_container_width=True, height=400)

            # Statistics
            st.subheader("📊 Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            total_timepoints = len(result.timepoints)
            evaluable = sum(1 for tp in result.timepoints if tp.current_response and tp.current_response != ResponseType.NOT_EVALUABLE)

            stat_col1.metric("Total Timepoints", total_timepoints)
            stat_col2.metric("Evaluable", evaluable)
            stat_col3.metric("CR Count", sum(1 for tp in result.timepoints if tp.confirmed_response == ResponseType.CR))
            stat_col4.metric("Progression Count", sum(1 for tp in result.timepoints if tp.confirmed_response == ResponseType.PROGRESSION))

            # Download button
            st.subheader("💾 결과 다운로드")
            excel_data = export_to_excel(result)
            st.download_button(
                label="📥 Excel로 다운로드",
                data=excel_data,
                file_name=f"{uploaded_file.name.replace('.xlsx', '')}_evaluated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Response chart
            st.subheader("📈 SPEP Trend")
            chart_df = df[df["SPEP"].notna()][["Date", "SPEP"]].copy()
            if not chart_df.empty:
                chart_df["Date"] = pd.to_datetime(chart_df["Date"])
                st.line_chart(chart_df.set_index("Date")["SPEP"])

    else:
        # Show instructions when no file is uploaded
        st.info("👈 왼쪽 사이드바에서 Excel 파일을 업로드하세요.")

        st.markdown("""
        ### 사용 방법
        1. 왼쪽 사이드바에서 **Excel 파일**을 업로드합니다.
        2. 파일에는 다음 데이터가 포함되어야 합니다:
           - **SPEP** (Monoclonal peak, M-protein)
           - **serum Kappa** (Kappa light chain)
           - **serum Lambda** (Lambda light chain)
           - UPEP (선택사항)
        """)

        st.markdown("---")
        st.markdown("### 평가 기준")

        # Patient Classification
        st.markdown("#### 환자 분류")
        st.markdown("""
        **1. IgG 타입** (SPEP ≥ 0.5 g/dL)
        - **IgG_Kappa**: Kappa > Lambda
        - **IgG_Lambda**: Lambda ≥ Kappa

        **2. LCD 타입** (SPEP < 0.5 g/dL AND |Kappa-Lambda| ≥ 100)
        - **LCD_Kappa**: Kappa > Lambda (iFLC = Kappa)
        - **LCD_Lambda**: Lambda > Kappa (iFLC = Lambda)

        **3. Unclassified**: SPEP < 0.5 AND |Kappa-Lambda| < 100
        """)

        st.markdown("---")

        # Response Evaluation
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### IgG 타입 반응 평가")
            st.markdown("""
            | 반응 | 기준 |
            |:----:|:-----|
            | **MR** | Baseline 대비 ≥15% 감소 |
            | **PR** | Baseline 대비 ≥45% 감소 |
            | **VGPR** | Baseline 대비 ≥85% 감소 |
            | **CR** | SPEP = 0 |
            | **Progression** | Nadir 대비 >0.45 g/dL 상승 |
            | **Progression (Type 변경!)** | CR 이후 \|Kappa-Lambda\| > 100 |
            """)

        with col2:
            st.markdown("#### LCD 타입 반응 평가")
            st.markdown("""
            | 반응 | 기준 |
            |:----:|:-----|
            | **CR** | FLC ratio 정상화 (0.26~1.65) |
            | **VGPR** | iFLC ≥90% 감소 또는 iFLC < 100 |
            | **PR** | iFLC ≥50% 감소 |
            | **PD** | iFLC ≥25% 증가 또는 절대 증가 ≥100 |
            """)
            st.caption("※ iFLC = involved FLC (LCD_Kappa→Kappa, LCD_Lambda→Lambda)")

        st.markdown("---")
        st.info("💡 **Note:** 반응 확정(Confirmed Response)은 **2회 연속** 동일한 반응이 필요합니다.")


if __name__ == "__main__":
    main()
