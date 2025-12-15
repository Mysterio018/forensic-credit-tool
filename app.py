import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai  # <--- NEW IMPORT

# --- 1. CONFIGURATION & BANK-GRADE STYLING ---
st.set_page_config(
    page_title="AI Forensic Credit Assessment Tool",
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
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'TotalAssets', 'TotalDebt', 
                'Equity', 'CurrentAssets', 'CurrentLiabilities', 'CFO', 'Interest', 
                'CFI', 'CFF', 'Capex', 'Inventory', 'Receivables', 'Cash']
        for c in cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 3. UNIFIED CALCULATION ENGINE (Handles Both Modes) ---
def calculate_metrics(df):
    if df.empty: return df
    
    # --- A. LIQUIDITY ANALYSIS ---
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    
    # --- B. PROFITABILITY ANALYSIS ---
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    # --- C. SOLVENCY & LEVERAGE ---
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # --- D. DUPONT INPUTS ---
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # --- E. EARNINGS QUALITY & ACCRUALS ---
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    
    # --- F. FORENSIC INDICATORS (BENEISH PROXY) ---
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # --- G. ALTMAN Z-SCORE (Emerging Market Model) ---
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # --- H. PIOTROSKI F-SCORE (SIMPLIFIED) ---
    df['F1'] = (df['ROA'] > 0).astype(int)
    df['F2'] = (df['CFO'] > 0).astype(int)
    df['F3'] = (df['CFO'] > df['PAT']).astype(int)
    df['F_Score'] = df['F1'] + df['F2'] + df['F3']
    
    # --- I. CASH FLOW ADEQUACY ---
    df['CF_Debt_Service'] = df['CFO'] / df['TotalDebt'].replace(0, 1)
    df['CF_Capex_Cov'] = df['CFO'] / df['Capex'].abs().replace(0, 1)

    # --- J. LIFE CYCLE STAGE ---
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # --- K. COMPOSITE CREDIT SCORE (0-100) ---
    def get_credit_score(row):
        score = 100
        if row['Z_Score'] < 1.23: score -= 25
        elif row['Z_Score'] < 2.9: score -= 10
        if row['CFO_to_PAT'] < 0.8: score -= 15
        if row['Debt_Equity'] > 2.0: score -= 15
        if row['Current_Ratio'] < 1.0: score -= 10
        if row['ICR'] < 1.5: score -= 10
        if row['Beneish_Flag_DSRI'] == 1: score -= 10
        return max(0, score)

    df['Credit_Score'] = df.apply(get_credit_score, axis=1)
    return df

# --- 4. AI & RULE-BASED GENERATORS (The Brains) ---

def get_rule_based_memo(row, company):
    """Generates the standard memo without needing an API key."""
    score = row['Credit_Score']
    
    if score >= 75: bucket, color, action = "LOW RISK", "#00c04b", "✅ APPROVE for Lending"
    elif score >= 50: bucket, color, action = "MEDIUM RISK", "#ffa700", "⚠️ APPROVE WITH CAUTION"
    else: bucket, color, action = "HIGH RISK", "#ff4b4b", "⛔ REJECT / SENIOR REVIEW"
        
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Bankruptcy Risk (Z: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Weak Earnings Quality (Paper Profits detected)")
    if row['Debt_Equity'] > 2.0: flags.append(f"High Leverage (D/E: {row['Debt_Equity']:.2f})")
    if row['Current_Ratio'] < 1.0: flags.append("Liquidity Stress (Current Ratio < 1.0)")
    
    flag_text = "\\n".join([f"- {f}" for f in flags]) if flags else "- No major forensic red flags detected."
    
    summary = f\"\"\"
    **Analysis Mode:** ⚙️ Standard Algorithm (Rule-Based)
    
    **1. Credit Assessment:**
    The borrower **{company}** falls into the **{bucket}** category (Score: {int(score)}/100).
    
    **2. Forensic Findings:**
    {flag_text}
    
    **3. Recommendation:**
    {action} based on quantitative thresholds.
    \"\"\"
    return bucket, action, summary, color

def get_gemini_memo(row, company, api_key):
    """Generates a custom memo using Google Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f\"\"\"
        Act as a senior credit underwriter at a bank. Write a formal credit memo for **{company}** based on these financials:
        
        **Financial Snapshot:**
        - Revenue: {row['Revenue']} | Net Profit: {row['PAT']} | CFO: {row['CFO']}
        - Total Debt: {row['TotalDebt']} | Equity: {row['Equity']}
        
        **Risk Indicators:**
        - Z-Score: {row['Z_Score']:.2f} (Distress < 1.23)
        - Debt/Equity: {row['Debt_Equity']:.2f}
        - CFO/PAT Ratio: {row['CFO_to_PAT']:.2f}
        
        **Task:**
        1. Determine the Risk Profile (Low/Medium/High).
        2. Identify the top 2 strengths and top 2 risks.
        3. Specifically comment on Earnings Quality (CFO vs PAT).
        4. Give a final "Approve" or "Reject" recommendation with a short reason.
        
        Keep it professional, concise, and structured.
        \"\"\"
        response = model.generate_content(prompt)
        
        # We define generic colors for AI mode since AI decides the risk
        return "AI ANALYZED", "🤖 SEE MEMO BELOW", f"**Analysis Mode:** 🧠 Live AI (Gemini Pro)\\n\\n{response.text}", "#4e8cff"
        
    except Exception as e:
        return "ERROR", "API ERROR", f"⚠️ API Failed: {str(e)}\nUsing Rule-Based Logic instead.", "#ff4b4b"

# --- 5. MAIN UI LAYOUT ---
def main():
    # Sidebar: Banker's Panel
    st.sidebar.title("🏦 Credit Underwriter")
    
    # --- INPUTS ---
    mode = st.sidebar.radio("Data Source:", ["📂 Select from Dataset", "✍️ Manual Data Entry"])
    
    # NEW: API KEY INPUT
    with st.sidebar.expander("🤖 AI Configuration"):
        api_key = st.text_input("Gemini API Key (Optional)", type="password", help="Enter key for Live AI analysis. Leave blank for standard mode.")

    st.sidebar.markdown("---")
    
    row = None
    
    if mode == "📂 Select from Dataset":
        raw_df = load_dataset()
        if raw_df.empty:
            st.error("Data missing! Please upload 'financials_master.csv'.")
            st.stop()
        
        company = st.sidebar.selectbox("Select Borrower", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("Select FY", years)
        st.sidebar.caption("Source: Audited Annual Statements")
        
        if st.sidebar.button("Run Forensic Analysis"):
            df_proc = calculate_metrics(raw_df)
            row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]
            
    else:
        with st.sidebar.form("manual"):
            st.subheader("Enter Financials")
            company_input = st.text_input("Company", "New Borrower Ltd")
            rev = st.number_input("Revenue", 10000.0)
            pat = st.number_input("PAT", 1500.0)
            cfo = st.number_input("CFO", 1200.0)
            debt = st.number_input("Total Debt", 5000.0)
            equity = st.number_input("Equity", 8000.0)
            ca = st.number_input("Current Assets", 6000.0)
            cl = st.number_input("Current Liab", 4000.0)
            ebit = st.number_input("EBIT", 2000.0)
            interest = st.number_input("Interest", 500.0)
            ta = st.number_input("Total Assets", 15000.0)
            rec = st.number_input("Receivables", 2000.0)
            
            if st.form_submit_button("Run Analysis"):
                data = {
                    'Company': [company_input], 'Year': [2025], 'Revenue': [rev], 'PAT': [pat],
                    'CFO': [cfo], 'TotalDebt': [debt], 'Equity': [equity], 'CurrentAssets': [ca],
                    'CurrentLiabilities': [cl], 'EBIT': [ebit], 'Interest': [interest],
                    'TotalAssets': [ta], 'Receivables': [rec], 'EBITDA': [ebit+500], 'CFI': [-500], 'CFF': [-200], 'Capex': [-300]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    # --- DASHBOARD ---
    if row is not None:
        st.title(f"🏢 Credit Report: {row['Company']}")
        
        # Tabs
        tabs = st.tabs(["1️⃣ Overview", "2️⃣ Financial Analysis", "3️⃣ DuPont & Quality", "4️⃣ Forensic & Manipulation", "5️⃣ Distress & EWS", "6️⃣ Cash Flow", "7️⃣ Credit Decision (AI)"])
        
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
            c3.metric("Total Debt", f"₹{row['TotalDebt']:,.0f}")
            c4.metric("Risk Score", f"{int(row['Credit_Score'])}", delta_color="off")
            st.markdown("---")
            # Determine initial bucket for display
            score = row['Credit_Score']
            b_display = "LOW RISK" if score > 75 else "MEDIUM RISK" if score > 50 else "HIGH RISK"
            st.info(f"**Calculated Risk Category:** {b_display}")

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

        with tabs[2]:
            st.subheader("🔗 A. DuPont Analysis")
            dupont = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']]
            })
            st.bar_chart(dupont.set_index('Driver'))
            st.subheader("🔍 B. Earnings Quality")
            q1, q2 = st.columns(2)
            q1.metric("CFO / PAT Ratio", f"{row['CFO_to_PAT']:.2f}", "Target > 0.8")
            q2.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}", "Lower is Better")

        with tabs[3]:
            st.subheader("🚩 A. Forensic Red Flags")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**1. Cash vs Profit**")
                if row['CFO'] < row['PAT']: st.error("❌ CFO < PAT (Weak Earnings Quality)")
                else: st.success("✅ CFO > PAT (Strong Earnings)")
            with c2:
                st.markdown("**2. Beneish Proxy**")
                if row['Beneish_Flag_DSRI'] == 1: st.error("❌ Receivables growing faster than Sales")
                else: st.success("✅ Revenue growth aligns with Receivables")

        with tabs[4]:
            st.subheader("📉 A. Altman Z-Score")
            st.metric("Z-Score", f"{row['Z_Score']:.2f}")
            st.progress(min(max(row['Z_Score']/5, 0), 1))
            st.caption("Distress < 1.23 | Safe > 2.9")
            st.subheader("⚠️ B. Early Warning Signals")
            e1, e2 = st.columns(2)
            e1.metric("Leverage Trend", "High" if row['Debt_Equity'] > 2 else "Stable")
            e2.metric("Liquidity", "Weak" if row['Current_Ratio'] < 1 else "Adequate")

        with tabs[5]:
            st.subheader("🔄 A. Cash Flow Structure")
            st.bar_chart(pd.DataFrame({'Type': ['CFO', 'CFI', 'CFF'], 'Value': [row['CFO'], row['CFI'], row['CFF']]}).set_index('Type'))
            st.subheader("🔄 B. Business Life-Cycle")
            st.info(f"📍 **Identified Stage:** {row['Life_Cycle']}")

        # TAB 7: CREDIT DECISION (THE AI PART)
        with tabs[6]:
            st.subheader("🤖 Smart Credit Decision")
            
            # --- LOGIC TO SWITCH BETWEEN RULE-BASED AND GEMINI ---
            if api_key:
                # Use Gemini if key is provided
                bucket, action, summary_text, color = get_gemini_memo(row, row['Company'], api_key)
            else:
                # Use Rule-Based if no key
                bucket, action, summary_text, color = get_rule_based_memo(row, row['Company'])
            
            st.markdown(f"""
            <div class="report-box" style="border-left: 5px solid {color};">
                <h2>{action}</h2>
                <h4>Risk Profile: <span style="color:{color}">{bucket}</span></h4>
                <hr>
                <pre style="white-space: pre-wrap; font-family: inherit;">{summary_text}</pre>
            </div>
            """, unsafe_allow_html=True)
            
            st.download_button("📩 Download Credit Memo", summary_text, file_name=f"Memo_{row['Company']}.txt")

    elif mode == "📂 Select from Dataset" and row is None:
        st.info("👈 Select company & click 'Run Forensic Analysis'")
    elif mode == "✍️ Manual Data Entry" and row is None:
        st.info("👈 Enter data & click 'Run Analysis'")

if __name__ == "__main__":
    main()
