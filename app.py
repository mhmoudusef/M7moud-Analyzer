import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytesseract
import cv2
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors

# ================== Page Config ==================
st.set_page_config(page_title="M7moud Analyzer", layout="wide")
st.title("🕒 M7moud Analyzer - Monthly Shift Analysis")

DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# ================= Session State =================
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# ================= TIME PARSING =================
def parse_shift(shift):
    if not isinstance(shift, str):
        return None
    shift = shift.strip().upper()
    if shift == "OFF" or "-" not in shift:
        return None
    try:
        start_str = shift.split("-")[0].strip()
        return datetime.strptime(start_str, "%H:%M").time()
    except:
        return None

# ================= CLASSIFICATION =================
def classify_shift(shift):
    start = parse_shift(shift)
    if start is None:
        return "OFF"

    if time(0,0) <= start < time(7,0):
        return "Over Night"
    if time(7,0) <= start < time(10,0):
        return "Morning"
    if time(10,0) <= start < time(15,0):
        return "Mid"
    if time(15,0) <= start <= time(23,59):
        return "Night"

    return "OFF"

# ================= ANALYSIS =================
def analyze_dataframe(df):
    result = []
    for _, row in df.iterrows():
        counts = {
            "Morning":0,
            "Mid":0,
            "Night":0,
            "Over Night":0,
            "OFF":0
        }
        for day in DAYS:
            if day in df.columns:
                shift_type = classify_shift(row[day])
                counts[shift_type] += 1

        result.append({
            "User": row["User"],
            "Name": row["Name"],
            **counts
        })
    return pd.DataFrame(result)

# ================= PDF GENERATION =================
def generate_pdf(total_df, night_df):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Monthly Shift Analysis Report", styles["Title"]))
    elements.append(Paragraph(" ", styles["Normal"]))

    elements.append(Paragraph("Monthly Total", styles["Heading2"]))
    table_data = [total_df.columns.tolist()] + total_df.values.tolist()
    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (2,1), (-1,-1), "CENTER")
    ]))
    elements.append(table)

    elements.append(Paragraph(" ", styles["Normal"]))
    elements.append(Paragraph("Night Shift Ranking", styles["Heading2"]))

    night_data = [night_df.columns.tolist()] + night_df.values.tolist()
    night_table = Table(night_data, repeatRows=1)
    night_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.black),
        ("ALIGN", (1,1), (-1,-1), "CENTER")
    ]))
    elements.append(night_table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# ================= UI =================
tab1, tab2 = st.tabs(["📊 Excel Upload", "📸 Image OCR"])

# ---------- EXCEL ----------
with tab1:

    st.info("📱 Mobile & 💻 Desktop supported (select multiple files at once)")

    files = st.file_uploader(
        "Upload Excel Files (Weekly)",
        type=["xlsx","xls"],
        accept_multiple_files=True,
        key="excel_upload"
    )

    if files:
        st.session_state.uploaded_files = files

    if st.session_state.uploaded_files:

        weekly_results = []

        for excel in st.session_state.uploaded_files:
            df = pd.read_excel(excel)

            st.markdown(f"### 📄 Raw Data – {excel.name}")
            st.dataframe(df, use_container_width=True)

            res = analyze_dataframe(df)
            weekly_results.append(res)

            st.markdown(f"### 📊 Weekly Result – {excel.name}")
            st.dataframe(res, use_container_width=True)
            st.bar_chart(res.set_index("Name")[["Morning","Mid","Night","Over Night"]])

        # ---------- Monthly Total ----------
        st.subheader("📊 Monthly Total")
        combined_df = pd.concat(weekly_results, ignore_index=True)
        total_df = combined_df.groupby(["User","Name"], as_index=False).sum()

        st.dataframe(total_df, use_container_width=True)
        st.bar_chart(total_df.set_index("Name")[["Morning","Mid","Night","Over Night"]])

        # ---------- Night Ranking ----------
        st.subheader("🌙 Night Shift Ranking (Most to Least)")
        night_ranking = total_df[["Name","Night"]].sort_values(by="Night", ascending=False).reset_index(drop=True)
        st.dataframe(night_ranking, use_container_width=True)

        # ---------- Downloads ----------
        excel_buffer = io.BytesIO()
        total_df.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)

        st.download_button(
            "⬇️ Download Excel Report",
            excel_buffer,
            "monthly_shift_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

       

# ---------- OCR ----------
with tab2:
    images = st.file_uploader(
        "Upload Schedule Images",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )

    if images:
        for img in images:
            st.image(img, caption=img.name, use_container_width=True)
            file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            text = pytesseract.image_to_string(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            st.text_area("OCR Output", text, height=200)
            st.warning("⚠️ OCR for review only – Excel is recommended for final analysis")
