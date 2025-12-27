import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, time
import pytesseract
import cv2

# ================== Streamlit Page ==================
st.set_page_config(page_title="M7moud Analyzer", layout="wide")
st.title("🕒 M7moud Analyzer - Multi-Week / Monthly Analysis")

DAYS = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

# ================= TIME PARSING =================
def parse_shift(shift):
    if not isinstance(shift, str):
        return None, None
    shift = shift.strip().upper()
    if shift == "OFF" or "-" not in shift:
        return None, None
    try:
        start_str, _ = shift.split("-")
        start_str = start_str.strip()
        # تحويل 00:00 صباحًا و 12:00 ظهر حسب نظام 24 ساعة
        if start_str == "12:00":
            start_time = datetime.strptime(start_str, "%H:%M").time()
        else:
            start_time = datetime.strptime(start_str, "%H:%M").time()
        return start_time, None
    except:
        return None, None

# ================= CLASSIFICATION =================
def classify_shift(shift):
    start, _ = parse_shift(shift)
    if start is None:
        return "OFF"

    # Over Night: 00:00 → 06:59
    if time(0,0) <= start < time(7,0):
        return "Over Night"

    # Morning: 07:00 → 09:59
    if time(7,0) <= start < time(10,0):
        return "Morning"

    # Mid: 10:00 → 14:59
    if time(10,0) <= start < time(15,0):
        return "Mid"

    # Night: 15:00 → 23:59
    if time(15,0) <= start <= time(23,59):
        return "Night"

    return "Unknown"

# ================= ANALYSIS =================
def analyze_dataframe(df):
    result = []
    unknown_found = False
    for _, row in df.iterrows():
        counts = {"Morning":0,"Mid":0,"Night":0,"Over Night":0,"OFF":0,"Unknown":0}
        for day in DAYS:
            if day in df.columns:
                shift_type = classify_shift(row[day])
                if shift_type == "Unknown":
                    unknown_found = True
                counts[shift_type] += 1
        if sum(counts.values()) > 0:
            result.append({"User": row["User"], "Name": row["Name"], **counts})
    return pd.DataFrame(result), unknown_found

# ================= OCR =================
def ocr_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray)

# ================= UI =================
tab1, tab2 = st.tabs(["📊 Excel Upload", "📸 Image Upload"])

# ---------- EXCEL ----------
with tab1:
    excels = st.file_uploader(
        "Upload Excel Files (one file per week)",
        type=["xlsx","xls"],
        accept_multiple_files=True
    )

    if excels:
        weekly_results = []
        st.subheader("📄 Raw Data Per Week")
        for excel in excels:
            df = pd.read_excel(excel)
            st.markdown(f"### Week File: {excel.name}")
            st.dataframe(df, use_container_width=True)

            # تحليل كل أسبوع
            res, unknown_found = analyze_dataframe(df)
            weekly_results.append(res)
            st.markdown(f"### Analysis Result: {excel.name}")
            if unknown_found:
                st.warning(f"⚠️ يوجد شيفتات Unknown في {excel.name}")
            st.dataframe(res, use_container_width=True)
            st.bar_chart(res.set_index("Name")[["Morning","Mid","Night","Over Night","OFF","Unknown"]])

        # ---------- Monthly Total ----------
        st.subheader("📊 Monthly Total (All Weeks Combined)")
        combined_df = pd.concat(weekly_results)
        total_df = combined_df.groupby(["User","Name"]).sum().reset_index()
        st.dataframe(total_df, use_container_width=True)
        st.bar_chart(total_df.set_index("Name")[["Morning","Mid","Night","Over Night","OFF","Unknown"]])

        # ---------- Night Shift Ranking ----------
        st.subheader("🌙 Night Shift Ranking (Most to Least)")
        night_ranking = total_df[["Name","Night"]].sort_values(by="Night", ascending=False).reset_index(drop=True)
        st.dataframe(night_ranking, use_container_width=True)

        # ---------- Download Monthly Total ----------
        total_df.to_excel("monthly_total_shift_report.xlsx", index=False)
        with open("monthly_total_shift_report.xlsx","rb") as f:
            st.download_button("⬇️ Download Monthly Total Report", f, "monthly_total_shift_report.xlsx")

# ---------- IMAGES ----------
with tab2:
    images = st.file_uploader(
        "Upload Schedule Images",
        type=["png","jpg","jpeg"],
        accept_multiple_files=True
    )

    if images:
        for img in images:
            with st.expander(img.name):
                st.image(img, use_container_width=True)
                file_bytes = np.asarray(bytearray(img.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, 1)
                text = pytesseract.image_to_string(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
                st.text_area("📄 OCR Output (Review)", text, height=200)
                st.warning("⚠️ OCR may need review. الأفضل تحويل البيانات لجدول Excel قبل التحليل النهائي.")
