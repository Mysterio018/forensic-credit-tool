import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ======================================================
# 1. APP CONFIGURATION
# ======================================================
st.set_page_config(
    page_title="AI-Assisted Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 2. GLOBAL STYLING (ONLY CSS UPDATED)
# ======================================================
st.markdown("""
<style>
/* ================= GLOBAL ================= */
.stApp {
    background-color: #ffffff;
    color: #000000;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background-color: #f8f9fa;
    border-right: 1px solid #e9ecef;
}

/* Sidebar labels */
div[data-testid="stSidebar"] label {
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Selectbox */
div[data-testid="stSelectbox"] > div > div {
    background-color: #e6fffa !important;
    border: 1px solid #16a34a !important;
    color: #000000 !important;
}

/* Text Input */
div[data-testid="stTextInput"] > div > div {
    background-color: #e6fffa !important;
    border: 1px solid #16a34a !important;
}

/* Global input text */
input {
    color: #000000 !important;
    font-weight: 600 !important;
}

/* Dropdown */
div[role="listbox"] {
    background-color: #ffffff !important;
}
div[role="listbox"] div {
    color: #000000 !important;
}

/* ================= METRIC CARDS ================= */
div[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 15px;
    border-radius: 8px;
    border: 1px solid #e0e0e0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
div[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-weight: 700;
}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 20px;
    border-bottom: 2px solid #f0f0f0;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    background-color: transparent;
    color: #555;
    font-size: 15px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    color: #16a34a;
    border-bottom: 3px solid #16a34a;
}

/* ================= VERDICT BOX ================= */
.verdict-box {
    background-color: #f0fdf4;
    padding: 25px;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
}

/* =====================================================
   FIX: MANUAL ENTRY BLACK INPUT BOXES (EXPANDERS)
   ===================================================== */
section[data-testid="stSidebar"]
div[data-testid="stExpander"]
div[data-testid="stNumberInput"]
div[data-baseweb="input"] {
    background-color: #e6fffa !important;
    border: 1.5px solid #16a34a !important;
    color: #000000 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"]
input {
    background-color: #e6fffa !important;
    color: #000000 !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"]
button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border-left: 1px solid #16a34a !important;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"]
button:hover {
    background-color: #f0fdf4 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"] > details > summary {
    background-color: #f8fafc !important;
    border-radius: 6px;
    font-weight: 700;
    color: #000000 !important;
}

section[data-testid="stSidebar"]
div[data-testid="stExpander"] * {
    color: #000000 !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 3. DATA LOADING ENGINE
# ======================================================
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("financials_master.csv")
        cols = [
            'Revenue','EBITDA','EBIT','PAT','Interest',
            'TotalAssets','TotalDebt','Equity',
            'CurrentAssets','CurrentLiabilities',
            'Inventory','Receivables','Cash',
            'CFO','CFI','CFF','Capex'
        ]
        for c in cols:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# ======================================================
# 4. CALCULATION ENGINE
# ======================================================
def calculate_metrics(df):
    if df.empty:
        return df

    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    df['ROCE'] = (df['EBIT'] / (df['TotalDebt'] + df['Equity']).replace(0, 1)) * 100
    df['Debtor_Days'] = (df['Receivables'] / df['Revenue'].replace(0, 1)) * 365
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)

    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)

    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)

    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > df['Sales_Growth'] * 1.3).astype(int)

    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)

    def lifecycle(r):
        if r['CFO'] < 0 and r['CFI'] < 0 and r['CFF'] > 0: return "Introduction"
        if r['CFO'] > 0 and r['CFI'] < 0 and r['CFF'] > 0: return "Growth"
        if r['CFO'] > 0 and r['CFI'] < 0 and r['CFF'] < 0: return "Mature"
        if r['CFO'] < 0: return "Decline/Stress"
        return "Transition"

    df['Life_Cycle'] = df.apply(lifecycle, axis=1)

    def score(r):
        s = 100
        if r['Z_Score'] < 1.23: s -= 25
        elif r['Z_Score'] < 2.9: s -= 10
        if r['CFO_to_PAT'] < 0.8: s -= 15
        if r['Debt_Equity'] > 2.0: s -= 15
        if r['Current_Ratio'] < 1.0: s -= 10
        if r['ICR'] < 1.5: s -= 10
        if r['Beneish_Flag_DSRI'] == 1: s -= 10
        return max(0, s)

    df['Credit_Score'] = df.apply(score, axis=1)
    return df

# ======================================================
# 5. CREDIT MEMO
# ======================================================
def generate_formal_memo(row):
    score = row['Credit_Score']
    if score >= 75:
        return "APPROVE","LOW RISK","#16a34a","Strong financials. Recommended for approval.","No material forensic red flags."
    elif score >= 50:
        return "REVIEW","MEDIUM RISK","#d97706","Moderate risk. Covenants advised.","Some forensic indicators need monitoring."
    else:
        return "REJECT","HIGH RISK","#dc2626","High financial stress. Lending not advised.","Multiple forensic red flags detected."

# ======================================================
# 6. MAIN APP
# ======================================================
def main():
    st.sidebar.title("AI-Assisted Forensic Credit Assessment Tool")
    mode = st.sidebar.radio("Data Source", ["Select from Dataset", "Manual Data Entry"])
    row = None

    if mode == "Select from Dataset":
        df = load_dataset()
        company = st.sidebar.selectbox("Name", df['Company'].unique())
        year = st.sidebar.selectbox("Financial Year", sorted(df[df['Company']==company]['Year'].unique(), reverse=True))
        if st.sidebar.button("Run Analysis"):
            row = calculate_metrics(df).query("Company==@company and Year==@year").iloc[0]

    else:
        with st.sidebar.form("manual_entry"):
            name = st.text_input("Name","New Applicant")
            with st.expander("Profit & Loss",True):
                rev = st.number_input("Revenue",10000.0)
                ebit = st.number_input("EBIT",2000.0)
                pat = st.number_input("Net Profit (PAT)",1500.0)
                interest = st.number_input("Interest Expense",500.0)
            with st.expander("Balance Sheet"):
                ta = st.number_input("Total Assets",15000.0)
                debt = st.number_input("Total Debt",5000.0)
                equity = st.number_input("Equity",8000.0)
                ca = st.number_input("Current Assets",6000.0)
                cl = st.number_input("Current Liabilities",4000.0)
                rec = st.number_input("Receivables",2000.0)
            with st.expander("Cash Flow"):
                cfo = st.number_input("CFO",-50.0)
                cfi = st.number_input("CFI",-500.0)
                cff = st.number_input("CFF",-200.0)
                capex = st.number_input("Capex",-300.0)

            if st.form_submit_button("Run Analysis"):
                df = pd.DataFrame({
                    'Company':[name],'Year':[2025],'Revenue':[rev],'EBIT':[ebit],
                    'PAT':[pat],'Interest':[interest],'TotalAssets':[ta],
                    'TotalDebt':[debt],'Equity':[equity],'CurrentAssets':[ca],
                    'CurrentLiabilities':[cl],'Receivables':[rec],
                    'CFO':[cfo],'CFI':[cfi],'CFF':[cff],'Capex':[capex]
                })
                row = calculate_metrics(df).iloc[0]

    if row is not None:
        st.markdown(f"## Credit Report — {row['Company']}")
        verdict, risk, color, rec, forensic = generate_formal_memo(row)
        st.markdown(f"<div class='verdict-box'><b style='color:{color}'>{verdict}</b><br>Risk: {risk} | Score: {int(row['Credit_Score'])}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
