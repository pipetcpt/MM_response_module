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
from myeloma_response.parser import ExcelParser, LabData
from myeloma_response.classifier import PatientClassifier, PatientType
from myeloma_response.evaluator import ResponseEvaluator, ResponseType, TreatmentSegment
from datetime import datetime


def parse_uploaded_file(uploaded_file):
    """Parse an uploaded Excel file and return lab_data and classification."""
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

        return lab_data, classification, None
    except Exception as e:
        return None, None, str(e)
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def evaluate_with_options(
    lab_data: LabData,
    classification,
    treatment_changes: list[datetime] = None,
    type_overrides: dict[datetime, PatientType] = None
):
    """Evaluate with treatment changes and type overrides."""
    try:
        evaluator = ResponseEvaluator()
        result = evaluator.evaluate(
            lab_data,
            classification,
            treatment_changes=treatment_changes,
            type_overrides=type_overrides
        )
        return result, None
    except Exception as e:
        return None, str(e)


def evaluate_uploaded_file(uploaded_file):
    """Evaluate an uploaded Excel file (legacy function for compatibility)."""
    lab_data, classification, error = parse_uploaded_file(uploaded_file)
    if error:
        return None, error
    return evaluate_with_options(lab_data, classification)


def create_results_dataframe(result):
    """Convert evaluation results to a pandas DataFrame."""
    from myeloma_response.classifier import PatientType

    # Check if there are multiple segments
    has_segments = hasattr(result, 'segments') and len(result.segments) > 1

    data = []
    display_idx = 0
    for idx, tp in enumerate(result.timepoints):
        # Skip combined rows (they are merged into later evaluations)
        if hasattr(tp, 'is_combined') and tp.is_combined:
            continue
        display_idx += 1
        # Determine type for this timepoint (from segment if available)
        if has_segments and hasattr(tp, 'segment_id'):
            segment = result.segments[tp.segment_id] if tp.segment_id < len(result.segments) else None
            current_type = segment.patient_type if segment else result.patient_type
        else:
            current_type = result.patient_type

        is_lcd_type = current_type.is_lcd_type()
        is_igg_type = current_type.is_igg_type()

        # Calculate FLC ratio
        flc_ratio = None
        if tp.kappa is not None and tp.lambda_ is not None and tp.lambda_ != 0:
            flc_ratio = round(tp.kappa / tp.lambda_, 2)

        # Calculate iFLC, uFLC, dFLC based on patient type
        iflc = None
        uflc = None
        dflc = None

        if tp.kappa is not None and tp.lambda_ is not None:
            dflc = round(abs(tp.kappa - tp.lambda_), 1)  # dFLC = |Kappa - Lambda|
            if is_lcd_type:
                # For LCD_Kappa: iFLC=Kappa, uFLC=Lambda
                # For LCD_Lambda: iFLC=Lambda, uFLC=Kappa
                is_kappa_type = current_type == PatientType.LCD_KAPPA
                iflc = round(tp.kappa, 1) if is_kappa_type else round(tp.lambda_, 1)
                uflc = round(tp.lambda_, 1) if is_kappa_type else round(tp.kappa, 1)

        row = {
            "Timepoint": display_idx,
            "Date": tp.date.strftime("%Y-%m-%d") if tp.date else None,
        }

        # Add segment info if multiple segments exist
        if has_segments and hasattr(tp, 'segment_id'):
            row["Segment"] = tp.segment_id + 1
            if hasattr(tp, 'is_new_baseline') and tp.is_new_baseline:
                row["Segment"] = f"{tp.segment_id + 1} (새 BL)"

        row["SPEP"] = tp.spep
        row["Kappa"] = tp.kappa
        row["Lambda"] = tp.lambda_
        row["FLC Ratio"] = flc_ratio
        row["dFLC"] = dflc  # |Kappa - Lambda|

        # Add iFLC and uFLC for LCD type patients
        if is_lcd_type:
            row["iFLC"] = iflc
            row["uFLC"] = uflc

        row["UPEP"] = tp.upep

        # Add type-specific columns with clear labels
        if is_lcd_type:
            row["%Change (iFLC from BL)"] = round(tp.percent_change_from_baseline, 1) if tp.percent_change_from_baseline is not None else None
            row["iFLC Nadir"] = tp.nadir_value
        elif is_igg_type:
            row["%Change (SPEP from BL)"] = round(tp.percent_change_from_baseline, 1) if tp.percent_change_from_baseline is not None else None
            row["SPEP Nadir"] = tp.nadir_value
        else:
            row["%Change"] = round(tp.percent_change_from_baseline, 1) if tp.percent_change_from_baseline is not None else None
            row["Nadir"] = tp.nadir_value

        # Add LCD warning to response display if present in notes
        lcd_warning = " (LCD Type 변경 확인!)" if tp.notes and "(LCD Type 변경 확인!)" in tp.notes else ""

        current_resp = tp.current_response.value if tp.current_response else None
        if current_resp and lcd_warning:
            current_resp = f"{current_resp}{lcd_warning}"
        row["Current Response"] = current_resp

        confirmed_resp = tp.confirmed_response.value if tp.confirmed_response else None
        if confirmed_resp and lcd_warning:
            confirmed_resp = f"{confirmed_resp}{lcd_warning}"
        row["Confirmed Response"] = confirmed_resp

        # Add Notes column for combination info and other notes
        row["Notes"] = tp.notes if tp.notes else None

        data.append(row)

    return pd.DataFrame(data)


def get_response_summary(result):
    """Get summary of key response dates."""
    best_response = None
    best_response_date = None
    cr_date = None
    progression_date = None

    # Filter out combined timepoints
    active_timepoints = [tp for tp in result.timepoints if not (hasattr(tp, 'is_combined') and tp.is_combined)]

    # Response order for comparison (best to worst for treatment response)
    response_order = [ResponseType.SD, ResponseType.MR, ResponseType.PR, ResponseType.VGPR, ResponseType.CR]

    for tp in active_timepoints:
        if tp.confirmed_response:
            if tp.confirmed_response == ResponseType.CR and cr_date is None:
                cr_date = tp.date
            # Track progression (including type change progression)
            if tp.confirmed_response in [ResponseType.PROGRESSION, ResponseType.PROGRESSION_TYPE_CHANGE] and progression_date is None:
                progression_date = tp.date
            # Only consider standard responses for best response
            if tp.confirmed_response in response_order:
                if best_response is None or (best_response in response_order and
                    response_order.index(tp.confirmed_response) > response_order.index(best_response)):
                    best_response = tp.confirmed_response
                    best_response_date = tp.date

    return {
        "best_response": best_response,
        "best_response_date": best_response_date,
        "cr_date": cr_date,
        "progression_date": progression_date
    }


def export_to_excel(result, classification=None, initial_type_override=None):
    """Export results to Excel bytes for download."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Summary sheet
        # Determine displayed patient type (considering override)
        if initial_type_override:
            displayed_type = initial_type_override
            auto_type = classification.patient_type.value if classification else "N/A"
        else:
            displayed_type = result.patient_type.value
            auto_type = displayed_type

        summary_data = {
            "Property": [
                "Auto Classification (자동 분류)",
                "Applied Type (적용 타입)",
                "Baseline SPEP",
                "Baseline Kappa",
                "Baseline Lambda"
            ],
            "Value": [
                auto_type,
                displayed_type,
                result.baseline_spep,
                result.baseline_kappa,
                result.baseline_lambda
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Add segments sheet if multiple segments exist
        if hasattr(result, 'segments') and len(result.segments) > 0:
            segment_data = []
            for seg in result.segments:
                override_text = " (수동 설정)" if seg.is_type_override else ""
                segment_data.append({
                    "Segment": seg.segment_id + 1,
                    "Start Date": seg.start_date.strftime("%Y-%m-%d"),
                    "Patient Type": f"{seg.patient_type.value}{override_text}",
                    "Baseline SPEP": seg.baseline_spep,
                    "Baseline Kappa": seg.baseline_kappa,
                    "Baseline Lambda": seg.baseline_lambda
                })
            pd.DataFrame(segment_data).to_excel(writer, sheet_name="Segments", index=False)

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
    치료 반응(nCR, VGPR, PR, MR, Progression)을 평가합니다.
    """)
    st.caption("※ Modified IMWG criteria for real-world data")

    # File upload
    st.sidebar.header("📁 파일 업로드")
    uploaded_file = st.sidebar.file_uploader(
        "Excel 파일을 선택하세요",
        type=['xlsx', 'xls'],
        help="SPEP, serum Kappa, serum Lambda 데이터가 포함된 Excel 파일"
    )

    if uploaded_file is not None:
        st.sidebar.success(f"✅ 파일 업로드됨: {uploaded_file.name}")

        # Parse file first
        with st.spinner("파일 분석 중..."):
            lab_data, classification, parse_error = parse_uploaded_file(uploaded_file)

        if parse_error:
            st.error(f"❌ 파일 파싱 오류: {parse_error}")
            return

        # Get available dates for selection
        available_dates = [d.strftime("%Y-%m-%d") for d in lab_data.dates]

        # Sidebar options for treatment changes and type overrides
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ 평가 옵션")

        # Initial type override
        st.sidebar.subheader("📋 초기 타입 설정")
        type_options = ["자동 분류", "IgG_Kappa", "IgG_Lambda", "LCD_Kappa", "LCD_Lambda"]
        initial_type = st.sidebar.selectbox(
            "환자 타입 (첫 번째 시점)",
            type_options,
            index=0,
            help="자동 분류를 선택하면 첫 검사값 기준으로 자동 분류됩니다."
        )

        # Treatment changes (re-baseline)
        st.sidebar.subheader("💊 치료 변경 시점")
        st.sidebar.caption("치료 변경 시 새로운 baseline으로 재평가합니다.")

        treatment_change_dates = []
        num_treatment_changes = st.sidebar.number_input(
            "치료 변경 횟수",
            min_value=0,
            max_value=10,
            value=0,
            key="num_treatment_changes"
        )

        for i in range(num_treatment_changes):
            tc_date = st.sidebar.selectbox(
                f"치료 변경 {i+1} 시점",
                available_dates[1:],  # Exclude first date
                key=f"tc_date_{i}"
            )
            if tc_date:
                treatment_change_dates.append(tc_date)

        # Type changes at specific dates
        st.sidebar.subheader("🔄 타입 변경 시점")
        st.sidebar.caption("특정 시점부터 환자 타입을 변경합니다.")

        type_overrides = {}
        num_type_changes = st.sidebar.number_input(
            "타입 변경 횟수",
            min_value=0,
            max_value=10,
            value=0,
            key="num_type_changes"
        )

        for i in range(num_type_changes):
            col1, col2 = st.sidebar.columns(2)
            with col1:
                change_date = st.selectbox(
                    f"시점 {i+1}",
                    available_dates[1:],  # Exclude first date
                    key=f"type_change_date_{i}"
                )
            with col2:
                change_type = st.selectbox(
                    f"타입",
                    ["IgG_Kappa", "IgG_Lambda", "LCD_Kappa", "LCD_Lambda"],
                    key=f"type_change_type_{i}"
                )
            if change_date:
                type_overrides[change_date] = change_type

        # Convert dates to datetime and types to PatientType
        treatment_changes_dt = []
        for d in treatment_change_dates:
            treatment_changes_dt.append(datetime.strptime(d, "%Y-%m-%d"))

        type_overrides_dt = {}
        type_map = {
            "IgG_Kappa": PatientType.IGG_KAPPA,
            "IgG_Lambda": PatientType.IGG_LAMBDA,
            "LCD_Kappa": PatientType.LCD_KAPPA,
            "LCD_Lambda": PatientType.LCD_LAMBDA,
        }

        for d, t in type_overrides.items():
            dt = datetime.strptime(d, "%Y-%m-%d")
            type_overrides_dt[dt] = type_map[t]

        # Handle initial type override
        if initial_type != "자동 분류":
            first_date = lab_data.dates[0]
            type_overrides_dt[first_date] = type_map[initial_type]

        # Evaluate with options
        with st.spinner("평가 중..."):
            result, error = evaluate_with_options(
                lab_data,
                classification,
                treatment_changes=treatment_changes_dt if treatment_changes_dt else None,
                type_overrides=type_overrides_dt if type_overrides_dt else None
            )

        if error:
            st.error(f"❌ 오류 발생: {error}")
        else:
            # Display results
            st.header("📊 분석 결과")

            # Patient classification
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("환자 분류")
                # Show initial auto classification
                st.metric("자동 분류", classification.patient_type.value)
                st.caption(classification.classification_reason)

                # Show applied type if different from auto classification
                if initial_type != "자동 분류":
                    st.metric("적용 타입", initial_type, delta="수동 설정", delta_color="off")
                elif result.segments and len(result.segments) > 0:
                    applied_type = result.segments[0].patient_type.value
                    if applied_type != classification.patient_type.value:
                        st.metric("적용 타입", applied_type, delta="수동 설정", delta_color="off")

            with col2:
                st.subheader("Baseline Values (초기)")
                baseline_col1, baseline_col2, baseline_col3, baseline_col4 = st.columns(4)
                baseline_col1.metric("SPEP", f"{result.baseline_spep:.2f}" if result.baseline_spep else "N/A")
                baseline_col2.metric("Kappa", f"{result.baseline_kappa:.2f}" if result.baseline_kappa else "N/A")
                baseline_col3.metric("Lambda", f"{result.baseline_lambda:.2f}" if result.baseline_lambda else "N/A")
                # Show FLC ratio
                if result.baseline_kappa and result.baseline_lambda and result.baseline_lambda != 0:
                    flc_ratio = result.baseline_kappa / result.baseline_lambda
                    ratio_status = "정상" if 0.26 <= flc_ratio <= 1.65 else "비정상"
                    baseline_col4.metric("FLC Ratio", f"{flc_ratio:.2f}", ratio_status)
                else:
                    baseline_col4.metric("FLC Ratio", "N/A")

            # Display segments info if there are multiple segments
            if hasattr(result, 'segments') and len(result.segments) > 1:
                st.subheader("📋 치료 구간 (Segments)")
                segment_data = []
                for seg in result.segments:
                    override_text = " (수동 설정)" if seg.is_type_override else ""
                    segment_data.append({
                        "구간": seg.segment_id + 1,
                        "시작일": seg.start_date.strftime("%Y-%m-%d"),
                        "타입": f"{seg.patient_type.value}{override_text}",
                        "Baseline SPEP": f"{seg.baseline_spep:.2f}" if seg.baseline_spep else "N/A",
                        "Baseline Kappa": f"{seg.baseline_kappa:.2f}" if seg.baseline_kappa else "N/A",
                        "Baseline Lambda": f"{seg.baseline_lambda:.2f}" if seg.baseline_lambda else "N/A"
                    })
                st.dataframe(pd.DataFrame(segment_data), use_container_width=True, hide_index=True)

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
                    st.metric("nCR Achieved", summary["cr_date"].strftime("%Y-%m-%d"))
                else:
                    st.metric("nCR Achieved", "Not achieved")

            with resp_col3:
                if summary["progression_date"]:
                    st.metric("Progression", summary["progression_date"].strftime("%Y-%m-%d"), delta="⚠️", delta_color="inverse")
                else:
                    st.metric("Progression", "None")

            # Serial results table
            st.subheader("📋 Serial Response Evaluation")
            df = create_results_dataframe(result)

            # Add CSS for horizontal scrolling
            st.markdown("""
            <style>
            .stDataFrame {
                overflow-x: auto !important;
            }
            .stDataFrame > div {
                overflow-x: auto !important;
            }
            </style>
            """, unsafe_allow_html=True)

            # Try to style the dataframe (requires jinja2)
            try:
                def highlight_response(val):
                    if val is None:
                        return ""
                    val_str = str(val)
                    # Check for LCD Type warning (gold background takes priority)
                    if "(LCD Type 변경 확인!)" in val_str:
                        return "background-color: #FFD700"  # Gold - LCD type check warning
                    elif val_str.startswith("nCR"):
                        return "background-color: #90EE90"  # Light green
                    elif val_str.startswith("VGPR"):
                        return "background-color: #98FB98"  # Pale green
                    elif val_str.startswith("PR"):
                        return "background-color: #FFFFE0"  # Light yellow
                    elif val_str.startswith("MR"):
                        return "background-color: #FAFAD2"  # Light goldenrod
                    elif val_str.startswith("Progression"):
                        return "background-color: #FFB6C1"  # Light pink
                    elif val_str == "NE":
                        return "background-color: #D3D3D3"  # Light gray
                    return ""

                def highlight_flc_ratio(val):
                    """Highlight FLC ratio when in normal range (0.26~1.65)."""
                    if val is not None and isinstance(val, (int, float)):
                        if 0.26 <= val <= 1.65:
                            return "background-color: #90EE90; font-weight: bold"  # Normal range - light green
                        elif val < 0.26 or val > 1.65:
                            return "background-color: #FFDAB9"  # Abnormal - peach
                    return ""

                def highlight_dflc(val):
                    """Highlight dFLC when > 100 (LCD type consideration)."""
                    if val is not None and isinstance(val, (int, float)):
                        if val > 100:
                            return "background-color: #FFD700; font-weight: bold"  # Gold - LCD type threshold
                    return ""

                def highlight_notes(val):
                    """Highlight notes for combination info."""
                    if val is None:
                        return ""
                    val_str = str(val)
                    if "[결합:" in val_str:
                        return "background-color: #E6F3FF"  # Light blue for combination info
                    return ""

                # Build style with available columns
                styled_df = df.style.map(
                    highlight_response,
                    subset=["Current Response", "Confirmed Response"]
                ).map(
                    highlight_flc_ratio,
                    subset=["FLC Ratio"]
                )

                # Add dFLC highlighting if column exists
                if "dFLC" in df.columns:
                    styled_df = styled_df.map(highlight_dflc, subset=["dFLC"])

                # Add notes highlighting
                styled_df = styled_df.map(highlight_notes, subset=["Notes"])

                st.dataframe(styled_df, use_container_width=True, height=400)
            except (ImportError, AttributeError):
                # Fallback: show without styling if jinja2 is not installed
                st.dataframe(df, use_container_width=True, height=400)

            # Add caption explaining the columns
            st.caption("※ dFLC = |Kappa - Lambda| (금색: >100, LCD 타입 기준)")
            if result.patient_type.is_lcd_type():
                st.caption("※ iFLC: involved FLC (LCD_Kappa→Kappa, LCD_Lambda→Lambda) | uFLC: uninvolved FLC")
                st.caption("※ %Change (iFLC from BL): iFLC의 Baseline 대비 변화율 | iFLC Nadir: iFLC 최저값 | FLC Ratio 정상범위: 0.26~1.65 (녹색)")
            elif result.patient_type.is_igg_type():
                st.caption("※ %Change (SPEP from BL): SPEP의 Baseline 대비 변화율 | SPEP Nadir: SPEP 최저값")
            st.caption("※ 3일 이내 측정된 값은 자동으로 결합되어 평가됩니다. [결합: ...]은 다른 날짜에서 가져온 값을 표시합니다.")

            # Statistics
            st.subheader("📊 Statistics")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

            # Filter out combined timepoints for statistics
            active_timepoints = [tp for tp in result.timepoints if not (hasattr(tp, 'is_combined') and tp.is_combined)]
            total_timepoints = len(active_timepoints)
            evaluable = sum(1 for tp in active_timepoints if tp.current_response and tp.current_response != ResponseType.NOT_EVALUABLE)

            stat_col1.metric("Total Timepoints", total_timepoints)
            stat_col2.metric("Evaluable", evaluable)
            stat_col3.metric("nCR Count", sum(1 for tp in active_timepoints if tp.confirmed_response == ResponseType.CR))
            stat_col4.metric("Progression Count", sum(1 for tp in active_timepoints if tp.confirmed_response == ResponseType.PROGRESSION))

            # Download button
            st.subheader("💾 결과 다운로드")
            # Pass classification and initial_type for proper export
            initial_type_for_export = initial_type if initial_type != "자동 분류" else None
            excel_data = export_to_excel(result, classification, initial_type_for_export)
            st.download_button(
                label="📥 Excel로 다운로드",
                data=excel_data,
                file_name=f"{uploaded_file.name.replace('.xlsx', '')}_evaluated.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Response chart - different based on patient type
            if result.patient_type.is_igg_type():
                st.subheader("📈 SPEP Trend")
                chart_df = df[df["SPEP"].notna()][["Date", "SPEP"]].copy()
                if not chart_df.empty:
                    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
                    st.line_chart(chart_df.set_index("Date")["SPEP"])

            elif result.patient_type.is_lcd_type():
                st.subheader("📈 FLC Trends")

                # Determine which is iFLC based on patient type
                is_kappa = result.patient_type == PatientType.LCD_KAPPA
                iflc_col = "Kappa" if is_kappa else "Lambda"
                iflc_label = "iFLC (Kappa)" if is_kappa else "iFLC (Lambda)"

                col_chart1, col_chart2 = st.columns(2)

                with col_chart1:
                    st.markdown(f"**{iflc_label} Trend**")
                    iflc_df = df[df[iflc_col].notna()][["Date", iflc_col]].copy()
                    if not iflc_df.empty:
                        iflc_df["Date"] = pd.to_datetime(iflc_df["Date"])
                        iflc_df = iflc_df.rename(columns={iflc_col: iflc_label})
                        st.line_chart(iflc_df.set_index("Date")[iflc_label])

                with col_chart2:
                    st.markdown("**FLC Ratio Trend**")
                    ratio_df = df[df["FLC Ratio"].notna()][["Date", "FLC Ratio"]].copy()
                    if not ratio_df.empty:
                        ratio_df["Date"] = pd.to_datetime(ratio_df["Date"])
                        st.line_chart(ratio_df.set_index("Date")["FLC Ratio"])
                        st.caption("정상 범위: 0.26 ~ 1.65")

            else:
                # Unclassified - show all trends
                st.subheader("📈 Lab Value Trends")
                chart_df = df[["Date", "SPEP", "Kappa", "Lambda"]].copy()
                chart_df = chart_df.dropna(subset=["Date"])
                if not chart_df.empty:
                    chart_df["Date"] = pd.to_datetime(chart_df["Date"])
                    st.line_chart(chart_df.set_index("Date"))

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

        **2. LCD 타입** (SPEP < 0.5 g/dL AND |Kappa-Lambda| > 100)
        - **LCD_Kappa**: Kappa > Lambda (iFLC = Kappa)
        - **LCD_Lambda**: Lambda > Kappa (iFLC = Lambda)

        **3. Unclassified**: SPEP < 0.5 AND |Kappa-Lambda| ≤ 100
        """)

        st.markdown("---")

        # Response Evaluation
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### IgG 타입 반응 평가")
            st.markdown("""
            | 반응 | 기준 |
            |:----:|:-----|
            | **MR** | Baseline 대비 ≥25% 감소 |
            | **PR** | Baseline 대비 ≥50% 감소 |
            | **VGPR** | Baseline 대비 ≥90% 감소 |
            | **nCR** | SPEP = 0 (near Complete Response) |
            | **PD** | Nadir 대비 ≥25% 증가 **AND** 절대 증가 ≥0.5 g/dL |
            | **LCD Type 변경 확인!** | \|Kappa-Lambda\| > 100 발견 시 |
            """)
            st.caption("※ SPEP 25% 증가만 충족 시 → SD (다른 증상 확인 필요!)")

        with col2:
            st.markdown("#### LCD 타입 반응 평가")
            st.markdown("""
            | 반응 | 기준 |
            |:----:|:-----|
            | **Progression (Type 변경!)** | SPEP ≥ 0.5 (IgG 타입 변경 가능성) |
            | **nCR** | FLC ratio 정상화 (0.26~1.65) |
            | **VGPR** | Baseline 대비 iFLC ≥90% 감소 또는 iFLC < 100 |
            | **PR** | Baseline 대비 iFLC ≥50% 감소 |
            | **PD** | Nadir 대비 iFLC ≥25% 증가 **AND** 절대 증가 ≥100 |
            """)
            st.caption("※ iFLC = involved FLC (LCD_Kappa→Kappa, LCD_Lambda→Lambda)")
            st.caption("※ iFLC 25% 증가만 충족 시 → SD (다른 증상 확인 필요!)")

        st.markdown("---")
        st.info("💡 **Note:** 반응 확정(Confirmed Response)은 **2회 연속** 동일한 반응이 필요합니다.")


if __name__ == "__main__":
    main()
