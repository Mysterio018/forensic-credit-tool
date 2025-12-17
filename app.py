import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import yfinance as yf

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI-Assisted Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Theme & Targeted CSS Fixes (The "Nuclear" White Text Fix)
st.markdown("""
    <style>
    /* =============================================
       MAIN THEME RESET
       ============================================= */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa; /* Light Sidebar Background */
        border-right: 1px solid #e9ecef;
    }
    
    /* =============================================
       SIDEBAR INPUTS - DARK MODE WITH WHITE TEXT
       ============================================= */

    /* 1. INPUT BOXES (Selectbox, Text, Number) */
    /* Target the container */
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="input"] > div {
        background-color: #262730 !important; /* Dark Grey Background */
        border: 1px solid #4a4a4a !important;
        color: white !important;
    }

    /* 2. FORCE TEXT COLOR INSIDE INPUTS */
    /* This wildcard forces EVERYTHING inside the box to be white */
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] [data-baseweb="input"] * {
        color: white !important;
        -webkit-text-fill-color: white !important;
        caret-color: white !important;
    }

    /* 3. DROPDOWN MENU - THE POPUP LIST (CRITICAL FIX) */
    /* This must be global, not scoped to stSidebar */
    ul[data-baseweb="menu"] {
        background-color: #262730 !important;
        border: 1px solid #444 !important;
    }
    
    /* The individual options in the list */
    li[data-baseweb="option"] {
        background-color: #262730 !important;
        color: white !important; /* Force text white */
    }
    
    /* The text inside the option div */
    li[data-baseweb="option"] div {
        color: white !important;
    }
    
    /* Hover state for options */
    li[data-baseweb="option"]:hover, li[data-baseweb="option"][aria-selected="true"] {
        background-color: #444444 !important;
        color: white !important;
    }

    /* 4. BUTTONS (Run Analysis) */
    [data-testid="stSidebar"] button {
        background-color: #262730 !important;
        border: 1px solid #4a4a4a !important;
    }
    /* Force button text to be white */
    [data-testid="stSidebar"] button p, 
    [data-testid="stSidebar"] button div {
        color: white !important;
    }
    [data-testid="stSidebar"] button:hover {
        background-color: #000000 !important;
        border-color: white !important;
    }

    /* 5. LABELS (Keep them dark on the light sidebar background) */
    [data-testid="stSidebar"] label {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    
    /* 6. NUMBER INPUT +/- BUTTONS */
    [data-testid="stSidebar"] button[data-testid="stNumberInputStepDown"],
    [data-testid="stSidebar"] button[data-testid="stNumberInputStepUp"] {
        color: white !important;
    }

    /* =============================================
       MAIN CONTENT STYLING
       ============================================= */
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricLabel"] { color: #555 !important; }
    div[data-testid="stMetricValue"] { color: #000 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { border-bottom: 2px solid #f0f0f0; }
    .stTabs [data-baseweb="tab"] { background-color: transparent; color: #555; }
    .stTabs [aria-selected="true"] { color: #008000; border-bottom: 3px solid #008000; }

    /* Verdict Box */
    .verdict-box {
        background-color: #f0fdf4; 
        padding: 25px;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #1f2937;
    }
    
    /* Global Text */
    p, h1, h2, h3, h4, h5, li, span, div { color: #000000; }
    /* The sidebar overrides above handle the white text exceptions */
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_dataset():
    try:
        # Read all as strings first to safely handle commas
        df = pd.read_csv("financials_master.csv", dtype=str)
        
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'Interest', 
                'TotalAssets', 'TotalDebt', 'Equity', 'CurrentAssets', 'CurrentLiabilities',
                'Inventory', 'Receivables', 'Cash',
                'CFO', 'CFI', 'CFF', 'Capex']
        
        # Clean and convert data
        for c in cols:
            if c not in df.columns: 
                df[c] = 0
            else:
                # Remove commas and convert to numeric
                df[c] = df[c].str.replace(',', '', regex=True)
                df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
        # Ensure Year is integer
        if 'Year' in df.columns:
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').fillna(0).astype(int)
            
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 3. YAHOO FINANCE DATA MAPPER ---
def get_yahoo_data(ticker_symbol):
    """
    Fetches live data from Yahoo Finance and maps it to our internal schema.
    """
    try:
        stock = yf.Ticker(ticker_symbol)
        
        # Yahoo returns dataframes with dates as columns
        bs = stock.balance_sheet          # Balance Sheet
        fin = stock.financials            # Income Statement
        cf = stock.cashflow               # Cash Flow
        
        if bs.empty or fin.empty:
            return None, "No financial data found for this ticker. Try adding suffix (e.g. '.NS' for India)."

        # Get the most recent year (first column)
        recent_date = bs.columns[0]
        year = recent_date.year
        
        # Helper to safely get value from Series
        def get_val(df, keys):
            for k in keys:
                if k in df.index:
                    val = df.loc[k, recent_date]
                    return float(val) if val is not None else 0.0
            return 0.0

        # MAPPING LOGIC (Yahoo Label -> Our Label)
        # Note: We divide by 10,000,000 (1e7) to convert absolute values to Crores
        data = {
            'Company': [f"{ticker_symbol.upper()} (Live)"],
            'Year': [year],
            'Revenue': [get_val(fin, ['Total Revenue', 'Operating Revenue']) / 1e7],
            'EBIT': [get_val(fin, ['EBIT', 'Operating Income']) / 1e7],
            'PAT': [get_val(fin, ['Net Income', 'Net Income Common Stockholders']) / 1e7],
            'Interest': [get_val(fin, ['Interest Expense', 'Interest Expense Non Operating']) / 1e7],
            
            'TotalAssets': [get_val(bs, ['Total Assets']) / 1e7],
            'TotalDebt': [get_val(bs, ['Total Debt', 'Total Liab', 'Long Term Debt']) / 1e7],
            'Equity': [get_val(bs, ['Total Stockholder Equity', 'Total Equity Gross Minorities']) / 1e7],
            'CurrentAssets': [get_val(bs, ['Current Assets', 'Total Current Assets']) / 1e7],
            'CurrentLiabilities': [get_val(bs, ['Current Liabilities', 'Total Current Liabilities']) / 1e7],
            'Receivables': [get_val(bs, ['Net Receivables', 'Receivables']) / 1e7],
            
            'CFO': [get_val(cf, ['Operating Cash Flow', 'Total Cash From Operating Activities']) / 1e7],
            'CFI': [get_val(cf, ['Investing Cash Flow', 'Total Cashflows From Investing Activities']) / 1e7],
            'CFF': [get_val(cf, ['Financing Cash Flow', 'Total Cash From Financing Activities']) / 1e7],
            'Capex': [get_val(cf, ['Capital Expenditure']) / 1e7] 
        }
        
        # Fallback for Total Debt if not explicitly present
        if data['TotalDebt'][0] == 0:
            short_debt = get_val(bs, ['Current Debt', 'Current Debt And Capital Lease Obligation']) / 1e7
            long_debt = get_val(bs, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation']) / 1e7
            data['TotalDebt'] = [short_debt + long_debt]

        return pd.DataFrame(data), None

    except Exception as e:
        return None, str(e)

# --- 4. UNIFIED CALCULATION ENGINE ---
def calculate_metrics(df):
    if df.empty: return df
    
    # Financial Analysis
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    df['ROCE'] = (df['EBIT'] / (df['TotalDebt'] + df['Equity']).replace(0, 1)) * 100
    df['Debtor_Days'] = (df['Receivables'] / df['Revenue'].replace(0, 1)) * 365
    
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # DuPont
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # Forensic
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # Distress (Altman Z''-Score)
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # Life Cycle
    df['CF_Debt_Cov'] = df['CFO'] / df['TotalDebt'].replace(0, 1)
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # Score
    def get_score(row):
        score = 100
        if row['Z_Score'] < 1.23: score -= 25
        elif row['Z_Score'] < 2.9: score -= 10
        if row['CFO_to_PAT'] < 0.8: score -= 15
        if row['Debt_Equity'] > 2.0: score -= 15
        if row['Current_Ratio'] < 1.0: score -= 10
        if row['ICR'] < 1.5: score -= 10
        if row['Beneish_Flag_DSRI'] == 1: score -= 10
        return max(0, score)
    df['Credit_Score'] = df.apply(get_score, axis=1)
    
    return df

# --- 5. CREDIT MEMO GENERATOR ---
def generate_formal_memo(row):
    score = row['Credit_Score']
    
    if score >= 75:
        verdict, risk_profile, color_hex = "APPROVE", "LOW RISK", "#008000"
        rec_text = "The borrower demonstrates strong financial health with robust liquidity and solvency metrics. Recommended for approval."
    elif score >= 50:
        verdict, risk_profile, color_hex = "REVIEW", "MEDIUM RISK", "#d97706"
        rec_text = "The borrower shows moderate risk. Financials are stable but require stricter covenants regarding leverage or working capital."
    else:
        verdict, risk_profile, color_hex = "REJECT", "HIGH RISK", "#dc2626"
        rec_text = "The borrower exhibits signs of significant financial distress. Lending is not recommended without substantial collateral or guarantees."

    forensic_notes = []
    if row['Z_Score'] < 1.23: forensic_notes.append("Altman Z''-Score indicates potential distress zone.")
    if row['CFO_to_PAT'] < 0.8: forensic_notes.append("Earnings Quality Concern: Operating Cash Flow is significantly lower than Net Profit.")
    if row['Beneish_Flag_DSRI'] == 1: forensic_notes.append("Revenue Recognition Alert: Receivables are growing faster than Revenue.")
    if row['Debt_Equity'] > 2.5: forensic_notes.append("Leverage Alert: Debt-to-Equity ratio exceeds conservative thresholds.")
    
    if not forensic_notes:
        forensic_text = "No material forensic red flags detected."
    else:
        forensic_text = "\n".join([f"- {note}" for note in forensic_notes])

    return verdict, risk_profile, color_hex, rec_text, forensic_text

# --- 6. MAIN UI ---
def main():
    st.sidebar.title("AI-Assisted Forensic Credit Assessment Tool")
    
    # ADDED YAHOO FINANCE TO THE OPTIONS
    mode = st.sidebar.radio("Data Source", ["Live Data (Yahoo Finance)", "Select from Dataset", "Manual Data Entry"])
    
    row = None
    
    # --- 1. YAHOO FINANCE LOGIC ---
    if mode == "Live Data (Yahoo Finance)":
        st.sidebar.info("Enter Ticker (e.g., 'INFY.NS' for Infosys, 'AAPL' for Apple)")
        ticker = st.sidebar.text_input("Ticker Symbol", "RELIANCE.NS")
        
        if st.sidebar.button("Fetch & Analyze"):
            with st.spinner("Fetching data from Yahoo Finance..."):
                yahoo_df, error = get_yahoo_data(ticker)
                if error:
                    st.sidebar.error(f"Error: {error}")
                else:
                    df_proc = calculate_metrics(yahoo_df)
                    row = df_proc.iloc[0]

    # --- 2. DATASET LOGIC ---
    elif mode == "Select from Dataset":
        raw_df = load_dataset()
        if raw_df.empty:
            st.sidebar.error("Master Dataset not found.")
            st.stop()
        
        company = st.sidebar.selectbox("Name", raw_df['Company'].unique())
        
        if company:
            years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
            year = st.sidebar.selectbox("Financial Year", years)
        
            if st.sidebar.button("Run Analysis"):
                df_proc = calculate_metrics(raw_df)
                row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]

    # --- 3. MANUAL ENTRY LOGIC ---
    else:
        with st.sidebar.form("manual_entry"):
            st.subheader("Borrower Details")
            company_input = st.text_input("Name", "New Applicant")
            
            with st.expander("Profit & Loss (INR Cr)", expanded=True):
                rev = st.number_input("Revenue", value=10000.0, step=100.0, format="%.2f")
                ebit = st.number_input("EBIT", value=2000.0, step=100.0, format="%.2f")
                pat = st.number_input("Net Profit (PAT)", value=1500.0, step=100.0, format="%.2f")
                interest = st.number_input("Interest Expense", value=500.0, step=50.0, format="%.2f")
            
            with st.expander("Balance Sheet (INR Cr)", expanded=False):
                ta = st.number_input("Total Assets", value=15000.0, step=100.0, format="%.2f")
                debt = st.number_input("Total Debt", value=5000.0, step=100.0, format="%.2f")
                equity = st.number_input("Equity", value=8000.0, step=100.0, format="%.2f")
                ca = st.number_input("Current Assets", value=6000.0, step=100.0, format="%.2f")
                cl = st.number_input("Current Liab.", value=4000.0, step=100.0, format="%.2f")
                rec = st.number_input("Receivables", value=2000.0, step=100.0, format="%.2f")
            
            with st.expander("Cash Flow (INR Cr)", expanded=False):
                cfo = st.number_input("CFO (Operating)", value=1200.0, step=100.0, format="%.2f")
                cfi = st.number_input("CFI (Investing)", value=-500.0, step=100.0, format="%.2f")
                cff = st.number_input("CFF (Financing)", value=-200.0, step=100.0, format="%.2f")
                capex = st.number_input("Capex", value=-300.0, step=100.0, format="%.2f")
            
            if st.form_submit_button("Run Analysis"):
                data = {
                    'Company': [company_input], 'Year': [2025], 'Revenue': [rev], 'EBIT': [ebit], 
                    'PAT': [pat], 'Interest': [interest], 'TotalAssets': [ta], 'TotalDebt': [debt], 
                    'Equity': [equity], 'CurrentAssets': [ca], 'CurrentLiabilities': [cl], 
                    'Receivables': [rec], 'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff], 'Capex': [capex]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    # --- REPORT RENDERER (VERDICT FIRST STRUCTURE) ---
    if row is not None:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(f"## Credit Assessment: {row['Company']}")
            st.markdown(f"**FY:** {row['Year']} | **Analysis Type:** Forensic & Fundamental")
        with c2:
            st.metric("Risk Score", f"{int(row['Credit_Score'])} / 100")
        
        st.markdown("---")

        tabs = st.tabs([
            "📝 Executive Verdict", 
            "⚠️ Forensic & Distress", 
            "📊 Financial Analysis", 
            "💧 Cash Flow", 
        ])

        with tabs[0]:
            verdict, r_profile, color, rec, forensics = generate_formal_memo(row)
            st.markdown(f"""
                <div class="verdict-box" style="border-left: 5px solid {color};">
                    <h3 style="color: {color}; margin:0;">RECOMMENDATION: {verdict}</h3>
                    <p style="font-size: 16px; margin: 5px 0 0 0;"><strong>Risk Profile:</strong> {r_profile}</p>
                </div>
            """, unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Financial Assessment")
                st.info(rec)
            with col2:
                st.subheader("Forensic Flags")
                if "No material" in forensics: st.success(forensics)
                else: st.warning(forensics)
            
            st.subheader("Key Performance Indicators (In Cr)")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Revenue", f"{row['Revenue']:,.0f}")
            k2.metric("Net Profit", f"{row['PAT']:,.0f}")
            k3.metric("EBITDA Margin", f"{(row['EBIT'])/row['Revenue']*100:.1f}%" if row['Revenue'] else "N/A")
            k4.metric("Debt/Equity", f"{row['Debt_Equity']:.2f}x")

        with tabs[1]:
            st.subheader("1. Bankruptcy Prediction (Altman Z''-Score)")
            z = row['Z_Score']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = z,
                title = {'text': "Z''-Score", 'font': {'color': 'black'}},
                gauge = {
                    'axis': {'range': [None, 5]},
                    'bar': {'color': "black"},
                    'steps': [{'range': [0, 1.23], 'color': "#ffcccb"}, 
                              {'range': [1.23, 2.9], 'color': "#fff4cc"}, 
                              {'range': [2.9, 5], 'color': "#d4edda"}]
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(t=30, b=30, l=30, r=30))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 2. Earnings Quality")
                val = row['CFO_to_PAT']
                st.metric("CFO / PAT Ratio", f"{val:.2f}x")
                if val < 0.8: st.error("⚠️ **Alert:** Profits may be artificial (Low Cash Conversion).")
                else: st.success("✅ **Pass:** Profits are backed by cash.")
            
            with c2:
                st.markdown("### 3. Revenue Integrity")
                st.metric("Beneish DSRI Flag", "Triggered" if row['Beneish_Flag_DSRI'] else "Clean")
                if row['Beneish_Flag_DSRI'] == 1: st.error("⚠️ **Alert:** Receivables growing much faster than Sales.")
                else: st.success("✅ **Pass:** Revenue/Receivable growth is aligned.")

        with tabs[2]:
            st.subheader("DuPont Analysis (ROE Breakdown)")
            dupont_df = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turns', 'Leverage'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage']]
            })
            fig_dupont = px.bar(dupont_df, x='Driver', y='Value', text_auto='.2f', 
                                title=f"ROE: {row['ROE']:.1f}% Components", color='Driver')
            st.plotly_chart(fig_dupont, use_container_width=True)

            st.divider()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Current Ratio", f"{row['Current_Ratio']:.2f}")
            c2.metric("Interest Cov.", f"{row['ICR']:.2f}")
            c3.metric("Debtor Days", f"{row['Debtor_Days']:.0f}")
            c4.metric("ROCE", f"{row['ROCE']:.1f}%")

        with tabs[3]:
            st.subheader("Cash Flow Waterfall")
            cf_df = pd.DataFrame({
                'Type': ['Operations', 'Investing', 'Financing'],
                'Amount': [row['CFO'], row['CFI'], row['CFF']]
            })
            fig_cf = px.bar(cf_df, x='Type', y='Amount', color='Amount', color_continuous_scale='RdBu')
            st.plotly_chart(fig_cf, use_container_width=True)
            st.info(f"**Implied Business Phase:** {row['Life_Cycle']}")

if __name__ == "__main__":
    main()
