import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- 1. CONFIGURATION & BANK-GRADE STYLING ---
st.set_page_config(
    page_title="AI-Based Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS for Fintech UI
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
    /* Headers */
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
    /* Status Colors */
    .pass { color: #00c04b; font-weight: bold; }
    .fail { color: #ff4b4b; font-weight: bold; }
    .warn { color: #ffa700; font-weight: bold; }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
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

# --- 2. AUTHENTICATION (Backend or Sidebar) ---
try:
    # Try getting key from secrets first
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = None # Will prompt in sidebar if missing

# --- 3. DATA LOADING ENGINE ---
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("financials_master.csv")
        # Ensure all columns from blueprint exist
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'Interest', 
                'TotalAssets', 'TotalDebt', 'Equity', 'CurrentAssets', 'CurrentLiabilities',
                'Inventory', 'Receivables', 'Cash',
                'CFO', 'CFI', 'CFF', 'Capex']
        for c in cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 4. UNIFIED ANALYSIS ENGINE (The "Brain") ---
def calculate_metrics(df):
    if df.empty: return df
    
    # --- TAB 2: FINANCIAL STATEMENT ANALYSIS ---
    # Liquidity
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    
    # Profitability
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    # Solvency
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # --- TAB 3: DUPONT & EARNINGS QUALITY ---
    # DuPont Components
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # Earnings Quality
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    
    # --- TAB 4: FORENSIC MODELS (BENEISH PROXY) ---
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    # Flag: If Receivables grow 30% faster than Sales = Channel Stuffing Risk
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # --- TAB 5: DISTRESS (ALTMAN Z) & EWS ---
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # Early Warning Signals (EWS)
    df['EWS_Liquidity'] = (df['Current_Ratio'] < 1.0).astype(int)
    df['EWS_Leverage'] = (df['Debt_Equity'] > 2.0).astype(int)
    df['EWS_Interest'] = (df['ICR'] < 1.5).astype(int)
    
    # --- TAB 6: CASH FLOW & LIFE CYCLE ---
    df['CF_Debt_Cov'] = df['CFO'] / df['TotalDebt'].replace(0, 1)
    df['CF_Capex_Cov'] = df['CFO'] / df['Capex'].abs().replace(0, 1)
    
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # --- TAB 7: COMPOSITE CREDIT SCORE ---
    def get_score(row):
        score = 100
        # Deductions based on blueprint logic
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

# --- 5. AI GENERATORS (Rule-Based & Gemini) ---
def get_rule_based_summary(row, company):
    score = row['Credit_Score']
    if score >= 75: bucket, action, color = "LOW RISK", "✅ APPROVE", "#00c04b"
    elif score >= 50: bucket, action, color = "MEDIUM RISK", "⚠️ CAUTION", "#ffa700"
    else: bucket, action, color = "HIGH RISK", "⛔ REJECT", "#ff4b4b"
    
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Distress Risk (Z: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Poor Earnings Quality (CFO < PAT)")
    if row['Beneish_Flag_DSRI'] == 1: flags.append("Potential Revenue Manipulation (Receivables > Sales)")
    if row['EWS_Leverage'] == 1: flags.append("High Leverage Warning")
    
    flag_text = "\n".join([f"- {f}" for f in flags]) if flags else "- No critical forensic red flags."
    
    return f"""
    **Analysis Mode:** ⚙️ Standard Rules
    
    **1. Credit Assessment**
    Score: {int(score)}/100 ({bucket})
    
    **2. Forensic Findings**
    {flag_text}
    
    **3. Recommendation**
    {action} based on quantitative thresholds.
    """, color

def get_gemini_summary(row, company, key):
    genai.configure(api_key=key)
    
    # ROBUST MODEL SELECTOR: Tries newest first, falls back to older ones
    # This prevents 404 errors if a specific model version is deprecated
    models_to_try = ['gemini-1.5-flash', 'gemini-pro', 'gemini-1.0-pro']
    
    model = None
    last_error = ""
    
    prompt = f"""
    Act as a senior credit risk officer. Write a formal credit memo for **{company}**.
    
    **Financial Data:**
    - Revenue: {row['Revenue']} | Net Profit: {row['PAT']} | CFO: {row['CFO']}
    - Debt: {row['TotalDebt']} | Equity: {row['Equity']}
    
    **Forensic Indicators:**
    - Z-Score: {row['Z_Score']:.2f}
    - Earnings Quality (CFO/PAT): {row['CFO_to_PAT']:.2f}
    - Beneish Flag: {'Detected' if row['Beneish_Flag_DSRI']==1 else 'None'}
    - Life Cycle Stage: {row['Life_Cycle']}
    
    **Task:**
    1. Classify Risk (Low/Medium/High).
    2. Analyze Strengths vs Risks.
    3. Provide a Forensic Verdict.
    4. Final Recommendation (Approve/Reject).
    """

    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return f"**Analysis Mode:** 🧠 Live AI ({model_name})\n\n{response.text}", "#4e8cff"
        except Exception as e:
            last_error = str(e)
            continue # Try the next model
            
    # If all models fail
    return f"⚠️ AI Error: Could not connect to Google AI. (Error: {last_error})\n\n(Falling back to rules...)", "#ff4b4b"


# --- 6. MAIN UI LAYOUT ---
def main():
    st.sidebar.title("🏦 Forensic Credit Tool")
    
    # --- A. MODE SELECTION ---
    mode = st.sidebar.radio("Data Source:", ["📂 Dataset Mode", "✍️ Manual Calculator"])
    
    # --- API KEY HANDLING ---
    active_key = API_KEY
    if not active_key:
        with st.sidebar.expander("🤖 AI Configuration"):
            active_key = st.text_input("Gemini API Key", type="password")
    
    if active_key:
        st.sidebar.success("✅ AI Engine Connected")
    else:
        st.sidebar.warning("⚠️ Using Rule-Based Logic")

    st.sidebar.markdown("---")
    
    row = None
    
    # --- B. INPUTS ---
    if mode == "📂 Dataset Mode":
        raw_df = load_dataset()
        if raw_df.empty:
            st.error("⚠️ Data missing. Upload 'financials_master.csv'.")
            st.stop()
        
        company = st.sidebar.selectbox("Borrower", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("FY", years)
        
        if st.sidebar.button("Run Forensic Credit Analysis"):
            df_proc = calculate_metrics(raw_df)
            row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]
            
    else:
        # MANUAL CALCULATOR FORM (Matching Blueprint Variables)
        with st.sidebar.form("manual"):
            st.subheader("Financial Input")
            company_input = st.text_input("Company Name", "New Entity")
            
            st.markdown("📘 **P&L Statement**")
            rev = st.number_input("Revenue", 10000.0)
            ebit = st.number_input("EBIT", 2000.0)
            pat = st.number_input("PAT", 1500.0)
            interest = st.number_input("Interest", 500.0)
            
            st.markdown("📗 **Balance Sheet**")
            ta = st.number_input("Total Assets", 15000.0)
            debt = st.number_input("Total Debt", 5000.0)
            equity = st.number_input("Equity", 8000.0)
            ca = st.number_input("Current Assets", 6000.0)
            cl = st.number_input("Current Liab", 4000.0)
            rec = st.number_input("Trade Receivables", 2000.0)
            
            st.markdown("📙 **Cash Flow**")
            cfo = st.number_input("CFO (Operating)", 1200.0)
            cfi = st.number_input("CFI (Investing)", -500.0)
            cff = st.number_input("CFF (Financing)", -200.0)
            capex = st.number_input("Capex", -300.0)
            
            if st.form_submit_button("Run Forensic Credit Analysis"):
                data = {
                    'Company': [company_input], 'Year': [2025], 'Revenue': [rev], 'EBIT': [ebit], 
                    'PAT': [pat], 'Interest': [interest], 'TotalAssets': [ta], 'TotalDebt': [debt], 
                    'Equity': [equity], 'CurrentAssets': [ca], 'CurrentLiabilities': [cl], 
                    'Receivables': [rec], 'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff], 'Capex': [capex]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    # --- MAIN PANEL (7 TABS) ---
    if row is not None:
        st.title(f"🔍 Credit Report: {row['Company']}")
        
        tabs = st.tabs([
            "1️⃣ Overview", "2️⃣ Financials", "3️⃣ DuPont", 
            "4️⃣ Forensic", "5️⃣ Distress", "6️⃣ Cash Flow", "7️⃣ AI Decision"
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
            
            score = row['Credit_Score']
            b_text = "LOW RISK" if score > 75 else "MEDIUM" if score > 50 else "HIGH"
            b_color = "green" if score > 75 else "orange" if score > 50 else "red"
            st.markdown(f"### Risk Bucket: :{b_color}[{b_text}]")

        # TAB 2: FINANCIALS
        with tabs[1]:
            st.subheader("Core Ratio Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("#### A. Liquidity")
                st.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
                st.metric("OCF Ratio", f"{row['OCF_Ratio']:.2f}x")
            with col2:
                st.markdown("#### B. Profitability")
                st.metric("Net Margin", f"{row['NPM']:.1f}%")
                st.metric("ROE", f"{row['ROE']:.1f}%")
            with col3:
                st.markdown("#### C. Solvency")
                st.metric("Debt/Equity", f"{row['Debt_Equity']:.2f}x")
                st.metric("Interest Cover", f"{row['ICR']:.2f}x")

        # TAB 3: DUPONT
        with tabs[2]:
            st.subheader("DuPont Decomposition")
            dupont_df = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']]
            })
            st.bar_chart(dupont_df.set_index('Driver'))
            
            st.markdown("---")
            st.subheader("Earnings Quality")
            c1, c2 = st.columns(2)
            c1.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}", help="(PAT-CFO)/Assets")
            c2.metric("CFO / PAT", f"{row['CFO_to_PAT']:.2f}")

        # TAB 4: FORENSIC
        with tabs[3]:
            st.subheader("Forensic Accounting Models")
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### Beneish M-Score Proxy")
                flag = row['Beneish_Flag_DSRI']
                st.metric("Receivables vs Sales Growth", "⚠️ Flagged" if flag else "✅ Aligned")
                if flag: st.error("Receivables growing significantly faster than Revenue (Channel Stuffing Risk).")
            
            with c2:
                st.markdown("#### Revenue Quality")
                if row['CFO'] < row['PAT']: 
                    st.metric("Cash Realization", "⚠️ Weak")
                    st.warning("Paper Profits: Cash Flow is lower than reported Profit.")
                else: 
                    st.metric("Cash Realization", "✅ Strong")

        # TAB 5: DISTRESS
        with tabs[4]:
            st.subheader("Bankruptcy & Distress Prediction")
            st.metric("Altman Z-Score", f"{row['Z_Score']:.2f}")
            st.progress(min(max(row['Z_Score']/5, 0), 1))
            st.caption("Distress < 1.23 | Grey 1.23-2.9 | Safe > 2.9")
            
            st.subheader("Early Warning Signals (EWS)")
            e1, e2, e3 = st.columns(3)
            e1.metric("Liquidity Alert", "TRIGGERED" if row['EWS_Liquidity'] else "None")
            e2.metric("Leverage Alert", "TRIGGERED" if row['EWS_Leverage'] else "None")
            e3.metric("Debt Service Alert", "TRIGGERED" if row['EWS_Interest'] else "None")

        # TAB 6: CASH FLOW
        with tabs[5]:
            st.subheader("Cash Flow Profile")
            st.bar_chart(pd.DataFrame({'Flow': [row['CFO'], row['CFI'], row['CFF']]}, index=['Operating', 'Investing', 'Financing']))
            
            st.info(f"📍 **Business Life-Cycle Stage:** {row['Life_Cycle']}")
            
            st.metric("Debt Service Coverage (CFO/Debt)", f"{row['CF_Debt_Cov']:.2f}")

        # TAB 7: AI DECISION
        with tabs[6]:
            st.subheader("🤖 AI-Assisted Credit Decision")
            
            # Switch between Rule-Based and Gemini based on key presence
            if active_key:
                summary, color = get_gemini_summary(row, row['Company'], active_key)
            else:
                summary, color = get_rule_based_summary(row, row['Company'])
            
            st.markdown(f"""
            <div class="report-box" style="border-left: 5px solid {color};">
                <pre style="white-space: pre-wrap; font-family: inherit; color: white;">{summary}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button("📩 Download Credit Memo", summary, file_name=f"Memo_{row['Company']}.txt")

    elif mode == "📂 Dataset Mode":
        st.info("👈 Select a company from the sidebar to begin.")
    else:
        st.info("👈 Enter financial data and click 'Run Forensic Credit Analysis'.")

if __name__ == "__main__":
    main()
