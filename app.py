import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. CONFIGURATION & BANK-GRADE STYLING ---
st.set_page_config(
    page_title="AI Forensic Underwriter",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    /* Metric Cards */
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    /* Section Headers */
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #ffffff; }
    /* Report Box */
    .report-box {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    /* Risk Labels */
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa700; font-weight: bold; }
    .risk-low { color: #00c04b; font-weight: bold; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e1e1e;
        border-radius: 5px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e8cff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("financials_master.csv")
        # Ensure all necessary columns exist (fill missing with 0)
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'TotalAssets', 'TotalDebt', 
                'Equity', 'CurrentAssets', 'CurrentLiabilities', 'CFO', 'Interest', 
                'CFI', 'CFF']
        for c in cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 3. UNIFIED CALCULATION ENGINE (Handles Both Modes) ---
def calculate_metrics(df):
    if df.empty: return df
    
    # --- A. LIQUIDITY ---
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    
    # --- B. PROFITABILITY ---
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    # --- C. SOLVENCY ---
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # --- D. DUPONT INPUTS ---
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # --- E. FORENSIC INDICATORS ---
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    
    # --- F. ALTMAN Z-SCORE (Emerging Market Model) ---
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # --- G. PIOTROSKI F-SCORE (SIMPLIFIED) ---
    df['F1'] = (df['ROA'] > 0).astype(int)
    df['F2'] = (df['CFO'] > 0).astype(int)
    df['F3'] = (df['CFO'] > df['PAT']).astype(int)
    # Note: Shift logic removed for manual single-row entry to be robust
    df['F_Score'] = df['F1'] + df['F2'] + df['F3']
    
    # --- H. LIFE CYCLE STAGE ---
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # --- I. COMPOSITE CREDIT SCORE (0-100) ---
    def get_credit_score(row):
        score = 100
        # Deductions
        if row['Z_Score'] < 1.23: score -= 30
        elif row['Z_Score'] < 2.9: score -= 15
        
        if row['CFO_to_PAT'] < 0.8: score -= 15
        if row['Debt_Equity'] > 2.0: score -= 15
        if row['Current_Ratio'] < 1.0: score -= 10
        if row['ICR'] < 1.5: score -= 10
        return max(0, score)

    df['Credit_Score'] = df.apply(get_credit_score, axis=1)
    
    return df

# --- 4. AI CREDIT MEMO GENERATOR ---
def generate_credit_memo(row, company):
    score = row['Credit_Score']
    
    # Risk Categorization
    if score >= 75:
        bucket, color, action = "LOW RISK", "#00c04b", "✅ APPROVE for Lending"
    elif score >= 50:
        bucket, color, action = "MEDIUM RISK", "#ffa700", "⚠️ APPROVE WITH CAUTION"
    else:
        bucket, color, action = "HIGH RISK", "#ff4b4b", "⛔ REJECT / SENIOR REVIEW"
        
    # Flag Identification
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Bankruptcy Risk (Z-Score: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Weak Earnings Quality (Paper Profits detected)")
    if row['Debt_Equity'] > 2.0: flags.append(f"High Leverage (D/E: {row['Debt_Equity']:.2f})")
    if row['Current_Ratio'] < 1.0: flags.append("Liquidity Stress (Current Ratio < 1.0)")
    
    flag_text = "\\n".join([f"- {f}" for f in flags]) if flags else "- No major forensic red flags detected."
    
    summary = f\"\"\"
    **Borrower:** {company}
    **Composite Score:** {int(score)}/100
    
    **1. Credit Assessment:**
    The borrower falls into the **{bucket}** category. The financial health indicators suggest a {bucket.lower()} probability of default.
    
    **2. Forensic Findings:**
    {flag_text}
    
    **3. Recommendation:**
    {action}
    \"\"\"
    return bucket, action, summary, color

# --- 5. MAIN UI LAYOUT ---
def main():
    # Sidebar: Banker's Panel
    st.sidebar.title("🏦 Credit Underwriter")
    
    # --- SECTION A: MODE SELECTOR ---
    mode = st.sidebar.radio("Data Source:", ["📂 Select from Dataset", "✍️ Manual Data Entry"])
    st.sidebar.markdown("---")
    
    row = None # Initialize row
    
    # --- SECTION B: INPUT CONTROLS ---
    if mode == "📂 Select from Dataset":
        raw_df = load_dataset()
        if raw_df.empty:
            st.error("Data missing! Please upload 'financials_master.csv'.")
            st.stop()
        
        company = st.sidebar.selectbox("Select Borrower", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("Select FY", years)
        st.sidebar.caption("Source: Audited Annual Statements")
        
        # Action Button
        if st.sidebar.button("Analyze Credit Risk"):
            # Process Data
            df_processed = calculate_metrics(raw_df)
            row = df_processed[(df_processed['Company'] == company) & (df_processed['Year'] == year)].iloc[0]
            
    else:
        # Manual Entry Form
        with st.sidebar.form("manual_entry"):
            st.subheader("Financial Year Input")
            company_input = st.text_input("Company Name", "New Borrower Ltd")
            year_input = st.number_input("FY", 2025)
            
            st.markdown("### 📉 Profit & Loss")
            rev = st.number_input("Revenue", 10000.0)
            ebit = st.number_input("EBIT", 2000.0)
            pat = st.number_input("Net Profit (PAT)", 1500.0)
            interest = st.number_input("Interest Expense", 500.0)
            
            st.markdown("### ⚖️ Balance Sheet")
            ta = st.number_input("Total Assets", 15000.0)
            debt = st.number_input("Total Debt", 5000.0)
            equity = st.number_input("Total Equity", 8000.0)
            ca = st.number_input("Current Assets", 6000.0)
            cl = st.number_input("Current Liabilities", 4000.0)
            
            st.markdown("### 🔄 Cash Flow")
            cfo = st.number_input("CFO (Operating)", 1200.0)
            cfi = st.number_input("CFI (Investing)", -500.0)
            cff = st.number_input("CFF (Financing)", -200.0)
            
            submitted = st.form_submit_button("Analyze Credit Risk")
            
            if submitted:
                # Create DataFrame from input
                data = {
                    'Company': [company_input], 'Year': [year_input],
                    'Revenue': [rev], 'EBIT': [ebit], 'PAT': [pat], 'Interest': [interest],
                    'TotalAssets': [ta], 'TotalDebt': [debt], 'Equity': [equity],
                    'CurrentAssets': [ca], 'CurrentLiabilities': [cl],
                    'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff]
                }
                df_input = pd.DataFrame(data)
                df_processed = calculate_metrics(df_input)
                row = df_processed.iloc[0]

    # --- MAIN APPLICATION RENDERING ---
    if row is not None:
        bucket, action, summary, color = generate_credit_memo(row, row['Company'])
        
        st.title(f"🏢 Credit Report: {row['Company']}")
        st.markdown(f"**FY:** {row['Year']} | **Data Source:** {mode}")
        
        # --- TABBED STRUCTURE ---
        tabs = st.tabs([
            "🟦 Overview", 
            "🟩 Financial Analysis", 
            "🟨 DuPont & Quality", 
            "🟥 Forensic & Distress", 
            "🟪 Cash Flow & Life Cycle", 
            "🟧 Credit Decision"
        ])
        
        # TAB 1: OVERVIEW
        with tabs[0]:
            st.subheader("Credit Snapshot")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
            c3.metric("Total Debt", f"₹{row['TotalDebt']:,.0f}")
            c4.metric("Risk Score", f"{int(row['Credit_Score'])}/100", delta_color="off")
            
            st.markdown("---")
            st.info(f"**Risk Category:** {bucket}")

        # TAB 2: FINANCIAL ANALYSIS
        with tabs[1]:
            st.subheader("🔹 A. Liquidity Analysis")
            l1, l2 = st.columns(2)
            l1.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x", "Target > 1.0")
            l2.metric("OCF Ratio", f"{row['OCF_Ratio']:.2f}x", "Higher is Better")
            
            st.subheader("🔹 B. Profitability Analysis")
            p1, p2, p3 = st.columns(3)
            p1.metric("Net Profit Margin", f"{row['NPM']:.1f}%")
            p2.metric("ROA", f"{row['ROA']:.1f}%")
            p3.metric("ROE", f"{row['ROE']:.1f}%")
            
            st.subheader("🔹 C. Solvency Analysis")
            s1, s2 = st.columns(2)
            s1.metric("Debt-to-Equity", f"{row['Debt_Equity']:.2f}x", "Target < 2.0")
            s2.metric("Interest Coverage", f"{row['ICR']:.2f}x", "Target > 3.0")

        # TAB 3: DUPONT & QUALITY
        with tabs[2]:
            st.subheader("🔗 A. DuPont Analysis (ROE Breakdown)")
            dupont = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']],
                'Type': ['Input', 'Input', 'Input', 'Output']
            })
            fig = px.bar(dupont, x='Driver', y='Value', color='Type', text_auto='.2f', title="Drivers of Return on Equity")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 B. Earnings Quality")
            q1, q2 = st.columns(2)
            q1.metric("CFO / PAT Ratio", f"{row['CFO_to_PAT']:.2f}", "Target > 0.8")
            q2.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}", "Lower is Better")
            if row['CFO_to_PAT'] < 0.8:
                st.warning("⚠️ **Warning:** Profits are not backed by sufficient cash flow.")

        # TAB 4: FORENSIC & DISTRESS
        with tabs[3]:
            st.subheader("🚩 A. Forensic Red Flags")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. Cash vs Profit Check**")
                if row['CFO'] < row['PAT']:
                    st.error(f"❌ Red Flag: CFO (₹{row['CFO']}) < PAT (₹{row['PAT']})")
                else:
                    st.success("✅ Pass: Cash Flow supports Profits")
            with c2:
                st.markdown("**2. Leverage Check**")
                if row['Debt_Equity'] > 2.0:
                    st.error(f"❌ Red Flag: High Leverage ({row['Debt_Equity']:.2f}x)")
                else:
                    st.success("✅ Pass: Leverage within limits")
            
            st.markdown("---")
            st.subheader("📉 B. Altman Z-Score (Distress Prediction)")
            z = row['Z_Score']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = z,
                gauge = {'axis': {'range': [None, 5]}, 'bar': {'color': "white"},
                         'steps': [{'range': [0, 1.23], 'color': "#ff4b4b"}, {'range': [1.23, 2.9], 'color': "#ffa700"}, {'range': [2.9, 5], 'color': "#00c04b"}]}
            ))
            st.plotly_chart(fig_gauge, use_container_width=True)
            if row['Company'] == "Yes Bank": st.caption("ℹ️ Note: Z-Score applies differently to Banking firms.")

        # TAB 5: CASH FLOW & LIFE CYCLE
        with tabs[4]:
            st.subheader("🔄 A. Cash Flow Structure")
            cf_data = pd.DataFrame({
                'Source': ['Operating', 'Investing', 'Financing'],
                'Amount': [row['CFO'], row['CFI'], row['CFF']]
            })
            fig_cf = px.bar(cf_data, x='Source', y='Amount', color='Amount', title="Cash Flow Mix")
            st.plotly_chart(fig_cf, use_container_width=True)
            
            st.subheader("🔄 B. Business Life-Cycle (Dickinson Model)")
            st.info(f"📍 **Identified Stage:** {row['Life_Cycle']}")

        # TAB 6: CREDIT DECISION
        with tabs[5]:
            st.subheader("🤖 AI-Assisted Credit Decision")
            
            st.markdown(f"""
            <div class="report-box" style="border-left: 5px solid {color};">
                <h2>{action}</h2>
                <h4>Risk Profile: <span style="color:{color}">{bucket}</span></h4>
                <hr>
                <pre style="white-space: pre-wrap; font-family: inherit;">{summary}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button("📩 Download Credit Memo", summary, file_name=f"Credit_Memo_{row['Company']}.txt")

    elif mode == "📂 Select from Dataset" and row is None:
        st.info("👈 Select a company and year from the sidebar, then click 'Analyze Credit Risk'.")
    elif mode == "✍️ Manual Data Entry" and row is None:
        st.info("👈 Enter financial data in the sidebar form and click 'Analyze Credit Risk'.")

if __name__ == "__main__":
    main()
