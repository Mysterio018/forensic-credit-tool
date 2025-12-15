import os

app_code = """
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURATION & OPTIMIZATION ---
st.set_page_config(
    page_title="AI Forensic Underwriter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS (Dark Mode FinTech Style)
st.markdown(\"\"\"
    <style>
    .main { background-color: #0e1117; }
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .report-box {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444;
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa700; font-weight: bold; }
    .risk-low { color: #00c04b; font-weight: bold; }
    </style>
    \"\"\", unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE (Cached for Speed) ---
@st.cache_data
def load_data():
    try:
        # Load CSV and handle missing columns gracefully
        df = pd.read_csv("financials_master.csv")
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'TotalAssets', 'TotalDebt', 
                'Equity', 'CurrentAssets', 'CurrentLiabilities', 'CFO', 'Interest', 
                'CFI', 'CFF']
        
        # Ensure all columns exist, fill with 0 if missing
        for c in cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
            
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 3. ANALYTICAL ENGINE (All Syllabus Ratios) ---
def calculate_metrics(df):
    if df.empty: return df
    
    # A. LIQUIDITY
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    
    # B. PROFITABILITY
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    # C. SOLVENCY
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # D. DUPONT INPUTS
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # E. FORENSIC CHECKS
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    
    # F. ALTMAN Z-SCORE (Emerging Market)
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # G. PIOTROSKI F-SCORE (Simplified 5-point)
    df['F1'] = (df['ROA'] > 0).astype(int)
    df['F2'] = (df['CFO'] > 0).astype(int)
    df['F3'] = (df['CFO'] > df['PAT']).astype(int)
    df['F4'] = (df['Debt_Equity'] <= df.groupby('Company')['Debt_Equity'].shift(-1).fillna(100)).astype(int)
    df['F5'] = (df['Current_Ratio'] >= df.groupby('Company')['Current_Ratio'].shift(-1).fillna(0)).astype(int)
    df['F_Score'] = df['F1'] + df['F2'] + df['F3'] + df['F4'] + df['F5']
    
    return df

# --- 4. AI GENERATOR (Logic-Based) ---
def generate_ai_credit_note(row, company, year):
    risk_score = 0
    flags = []
    
    # Scoring Logic
    if row['Z_Score'] < 1.23: 
        risk_score += 3
        flags.append("High Bankruptcy Risk (Z-Score < 1.23)")
    if row['CFO_to_PAT'] < 0.8:
        risk_score += 2
        flags.append("Low Earnings Quality (Paper Profits)")
    if row['Debt_Equity'] > 2.0:
        risk_score += 2
        flags.append("High Leverage (D/E > 2.0)")
    if row['Current_Ratio'] < 1.0:
        risk_score += 2
        flags.append("Liquidity Stress (CR < 1.0)")

    # Decision
    if risk_score >= 4:
        bucket = "HIGH RISK"
        action = "⛔ REJECT / SENIOR REVIEW"
        color = "#ff4b4b" # Red
    elif risk_score >= 2:
        bucket = "MEDIUM RISK"
        action = "⚠️ CAUTION / STRICT COVENANTS"
        color = "#ffa700" # Orange
    else:
        bucket = "LOW RISK"
        action = "✅ APPROVE LOAN"
        color = "#00c04b" # Green
        
    return bucket, action, color, flags

# --- 5. MAIN UI LAYOUT ---
def main():
    # Load & Calc
    raw_df = load_data()
    if raw_df.empty:
        st.error("⚠️ Data missing! Please upload 'financials_master.csv' to the sidebar folder.")
        st.stop()
    
    df = calculate_metrics(raw_df)

    # Sidebar
    st.sidebar.title("🔍 AI Forensic Tool")
    company = st.sidebar.selectbox("Select Borrower", df['Company'].unique())
    year = st.sidebar.selectbox("Select FY", sorted(df[df['Company'] == company]['Year'].unique(), reverse=True))
    
    # Navigation
    page = st.sidebar.radio("Navigate to:", [
        "1️⃣ Company Overview",
        "2️⃣ Financial Analysis", 
        "3️⃣ DuPont & Quality",
        "4️⃣ Forensic & Fraud",
        "5️⃣ Cash Flow & Life Cycle",
        "6️⃣ Credit Decision (AI)"
    ])
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Active Profile:** {company} ({year})")

    # Get Row Data
    row = df[(df['Company'] == company) & (df['Year'] == year)].iloc[0]

    # --- PAGES ---
    
    # PAGE 1: OVERVIEW
    if page == "1️⃣ Company Overview":
        st.title(f"🏢 {company}: Executive Snapshot")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Revenue", f"₹{row['Revenue']:,.0f} Cr")
        col2.metric("Net Profit", f"₹{row['PAT']:,.0f} Cr")
        col3.metric("Total Debt", f"₹{row['TotalDebt']:,.0f} Cr")
        col4.metric("CFO", f"₹{row['CFO']:,.0f} Cr", delta_color="normal")
        
        st.markdown("### 📈 Revenue vs Profit Trend")
        chart_data = df[df['Company'] == company]
        fig = px.line(chart_data, x='Year', y=['Revenue', 'PAT', 'CFO'], markers=True)
        st.plotly_chart(fig, use_container_width=True)

    # PAGE 2: RATIOS
    elif page == "2️⃣ Financial Analysis":
        st.title("🧮 Financial Statement Analysis")
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Liquidity & Solvency")
            st.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x", "Target > 1.0")
            st.metric("Debt-to-Equity", f"{row['Debt_Equity']:.2f}x", "Target < 2.0")
            st.metric("Interest Coverage", f"{row['ICR']:.2f}x", "Target > 3.0")
            
        with c2:
            st.subheader("Profitability")
            st.metric("Net Profit Margin", f"{row['NPM']:.1f}%")
            st.metric("ROA", f"{row['ROA']:.1f}%")
            st.metric("ROE", f"{row['ROE']:.1f}%")

    # PAGE 3: DUPONT
    elif page == "3️⃣ DuPont & Quality":
        st.title("🧬 DuPont & Earnings Quality")
        
        # DuPont Chart
        dupont = pd.DataFrame({
            'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
            'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']],
            'Type': ['Input', 'Input', 'Input', 'Output']
        })
        fig = px.bar(dupont, x='Driver', y='Value', color='Type', text_auto='.2f', title="ROE Decomposition")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Earnings Quality Check")
        c1, c2 = st.columns(2)
        c1.metric("CFO / PAT Ratio", f"{row['CFO_to_PAT']:.2f}", "Target > 0.8")
        c2.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}", "Lower is Better")

    # PAGE 4: FORENSIC
    elif page == "4️⃣ Forensic & Fraud":
        st.title("🕵️ Forensic Red Flags")
        
        # Z-Score Gauge
        st.subheader("Altman Z-Score (Bankruptcy Risk)")
        z = row['Z_Score']
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = z,
            gauge = {'axis': {'range': [None, 5]}, 
                     'steps': [{'range': [0, 1.23], 'color': "#ff4b4b"}, 
                               {'range': [1.23, 2.9], 'color': "#ffa700"},
                               {'range': [2.9, 5], 'color': "#00c04b"}]}
        ))
        st.plotly_chart(fig, use_container_width=True)
        if company == "Yes Bank": st.info("ℹ️ Note: Z-Score applies differently to Banks.")
        
        st.markdown("---")
        st.subheader("Piotroski F-Score (Strength)")
        st.metric("F-Score", f"{int(row['F_Score'])} / 5", "Higher is Stronger")

    # PAGE 5: CASH FLOW
    elif page == "5️⃣ Cash Flow & Life Cycle":
        st.title("🔄 Cash Flow Analysis")
        
        cf_df = pd.DataFrame({
            'Type': ['Operating', 'Investing', 'Financing'],
            'Amount': [row['CFO'], row['CFI'], row['CFF']]
        })
        fig = px.bar(cf_df, x='Type', y='Amount', color='Amount', title="Cash Flow Mix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Life Cycle Logic
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: stage = "Introduction"
        elif cfo > 0 and cfi < 0 and cff > 0: stage = "Growth"
        elif cfo > 0 and cfi < 0 and cff < 0: stage = "Mature"
        elif cfo < 0: stage = "Decline/Stress"
        else: stage = "Transition"
        
        st.info(f"📍 **Business Life Cycle Stage:** {stage}")

    # PAGE 6: AI DECISION
    elif page == "6️⃣ Credit Decision (AI)":
        bucket, action, color, flags = generate_ai_credit_note(row, company, year)
        
        st.title("🤖 AI Credit Recommendation")
        
        st.markdown(f\"\"\"
        <div class="report-box" style="border-left: 5px solid {color};">
            <h2>{action}</h2>
            <h4>Risk Profile: <span style="color:{color}">{bucket}</span></h4>
            <hr>
            <p><strong>AI Assessment:</strong> The borrower {company} has been analyzed using forensic algorithms. 
            Based on FY{year} data, the system recommends the above action.</p>
        </div>
        \"\"\", unsafe_allow_html=True)
        
        if flags:
            st.subheader("🚨 Key Risk Drivers")
            for f in flags:
                st.error(f"• {f}")
        else:
            st.success("✅ No major forensic red flags detected.")

if __name__ == "__main__":
    main()
"""

with open("app.py", "w", encoding='utf-8') as f:
    f.write(app_code)

print("✅ App successfully created! Ready to launch.")