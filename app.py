import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- 1. CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Bank-Grade CSS
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
    /* Tabs styling */
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

# --- 3. METRICS CALCULATION ENGINE ---
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
    
    # E. FORENSIC INDICATORS
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    
    # Forensic Proxy: Sales vs Receivables Growth
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # F. ALTMAN Z-SCORE
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # G. PIOTROSKI F-SCORE
    df['F1'] = (df['ROA'] > 0).astype(int)
    df['F2'] = (df['CFO'] > 0).astype(int)
    df['F3'] = (df['CFO'] > df['PAT']).astype(int)
    df['F_Score'] = df['F1'] + df['F2'] + df['F3']
    
    # H. LIFE CYCLE
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # I. COMPOSITE SCORE
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

# --- 4. DUAL AI ENGINE (Rule-Based + Gemini) ---

def get_rule_based_summary(row, company):
    """Fallback logic if no API key is provided."""
    score = row['Credit_Score']
    if score >= 75: bucket, action = "LOW RISK", "✅ APPROVE"
    elif score >= 50: bucket, action = "MEDIUM RISK", "⚠️ CAUTION"
    else: bucket, action = "HIGH RISK", "⛔ REJECT"
    
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Bankruptcy Risk (Z: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Weak Earnings Quality")
    flag_text = "\n".join([f"- {f}" for f in flags]) if flags else "- No major red flags."
    
    return f"""
    **Analysis Mode:** Algorithm (Rule-Based)
    **Borrower:** {company}
    
    **1. Risk Assessment:**
    The borrower is categorized as **{bucket}** (Score: {int(score)}/100).
    
    **2. Forensic Findings:**
    {flag_text}
    
    **3. Recommendation:**
    {action} based on quantitative thresholds.
    """

def get_gemini_summary(row, company, api_key):
    """Live AI Analysis using Google Gemini."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a senior credit underwriter. Write a professional credit memo for {company} based on this data:
        
        Financials:
        - Revenue: {row['Revenue']}
        - Net Profit: {row['PAT']}
        - CFO: {row['CFO']}
        - Debt: {row['TotalDebt']}
        
        Risk Indicators:
        - Z-Score: {row['Z_Score']:.2f} (Safe > 2.9, Distress < 1.23)
        - Current Ratio: {row['Current_Ratio']:.2f}
        - Debt/Equity: {row['Debt_Equity']:.2f}
        - CFO/PAT Ratio: {row['CFO_to_PAT']:.2f}
        
        Provide:
        1. Executive Summary
        2. Key Risks (Bullet points)
        3. Forensic Verdict (Comment on Earnings Quality)
        4. Final Lending Decision (Approve/Reject)
        """
        response = model.generate_content(prompt)
        return f"**Analysis Mode:** 🧠 Live AI (Gemini Pro)\n\n" + response.text
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n\nFalling back to Rule-Based Summary..."

# --- 5. MAIN UI ---
def main():
    st.sidebar.title("🏦 Credit Underwriter")
    
    # --- INPUTS ---
    mode = st.sidebar.radio("Data Source:", ["📂 Select from Dataset", "✍️ Manual Data Entry"])
    
    # API Key Input (Secure)
    with st.sidebar.expander("🤖 AI Configuration"):
        api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.caption("Leave blank to use Rule-Based Logic.")

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
        
        tabs = st.tabs(["1️⃣ Overview", "2️⃣ Financials", "3️⃣ DuPont", "4️⃣ Forensic", "5️⃣ Distress", "6️⃣ Cash Flow", "7️⃣ AI Decision"])
        
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
            c3.metric("Debt", f"₹{row['TotalDebt']:,.0f}")
            c4.metric("Risk Score", f"{int(row['Credit_Score'])}", delta_color="off")
            st.info(f"**Calculated Risk Category:** {'HIGH' if row['Credit_Score']<50 else 'LOW' if row['Credit_Score']>75 else 'MEDIUM'}")

        with tabs[1]:
            c1, c2 = st.columns(2)
            c1.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
            c2.metric("Debt/Equity", f"{row['Debt_Equity']:.2f}x")
            st.metric("Net Profit Margin", f"{row['NPM']:.1f}%")

        with tabs[2]:
            st.subheader("DuPont ROE Breakdown")
            dupont = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']]
            })
            st.bar_chart(dupont.set_index('Driver'))

        with tabs[3]:
            st.subheader("Forensic Red Flags")
            c1, c2 = st.columns(2)
            with c1:
                if row['CFO'] < row['PAT']: st.error("❌ CFO < PAT (Weak Earnings Quality)")
                else: st.success("✅ CFO > PAT (Strong Earnings)")
            with c2:
                if row['Beneish_Flag_DSRI'] == 1: st.error("❌ Receivables growing faster than Sales")
                else: st.success("✅ Revenue growth aligns with Receivables")

        with tabs[4]:
            st.metric("Altman Z-Score", f"{row['Z_Score']:.2f}")
            st.progress(min(max(row['Z_Score']/5, 0), 1))
            st.caption("Distress < 1.23 | Safe > 2.9")

        with tabs[5]:
            st.info(f"📍 **Business Stage:** {row['Life_Cycle']}")
            st.bar_chart(pd.DataFrame({'Type': ['CFO', 'CFI', 'CFF'], 'Value': [row['CFO'], row['CFI'], row['CFF']]}).set_index('Type'))

        with tabs[6]:
            st.subheader("🤖 Smart Credit Decision")
            
            # Logic to switch between Gemini and Rule-Based
            if api_key:
                summary_text = get_gemini_summary(row, row['Company'], api_key)
            else:
                summary_text = get_rule_based_summary(row, row['Company'])
            
            st.markdown(f"""<div class="report-box">{summary_text}</div>""", unsafe_allow_html=True)
            st.download_button("📩 Download Credit Memo", summary_text, file_name=f"Memo_{row['Company']}.txt")

    elif mode == "📂 Select from Dataset" and row is None:
        st.info("👈 Select company & click 'Run Forensic Analysis'")
    elif mode == "✍️ Manual Data Entry" and row is None:
        st.info("👈 Enter data & click 'Run Analysis'")

if __name__ == "__main__":
    main()
