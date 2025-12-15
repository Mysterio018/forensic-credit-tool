import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

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
    .stMetric {
        background-color: #262730;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e8cff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; color: #ffffff; }
    .report-box {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa700; font-weight: bold; }
    .risk-low { color: #00c04b; font-weight: bold; }
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

# --- 3. UNIFIED CALCULATION ENGINE ---
def calculate_metrics(df):
    if df.empty: return df
    
    # A. Liquidity
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    
    # B. Profitability
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    
    # C. Solvency
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # D. DuPont
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # E. Forensic
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # F. Z-Score
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # G. Life Cycle
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # H. Composite Score
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

# --- 4. MEMO GENERATORS ---

def get_rule_based_memo(row, company):
    score = row['Credit_Score']
    if score >= 75: bucket, color, action = "LOW RISK", "#00c04b", "✅ APPROVE for Lending"
    elif score >= 50: bucket, color, action = "MEDIUM RISK", "#ffa700", "⚠️ APPROVE WITH CAUTION"
    else: bucket, color, action = "HIGH RISK", "#ff4b4b", "⛔ REJECT / SENIOR REVIEW"
        
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Bankruptcy Risk (Z: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Weak Earnings Quality (Paper Profits detected)")
    if row['Debt_Equity'] > 2.0: flags.append(f"High Leverage (D/E: {row['Debt_Equity']:.2f})")
    
    flag_text = "\n".join([f"- {f}" for f in flags]) if flags else "- No major forensic red flags detected."
    
    # NOTE: The triple quotes below are clean (no backslashes)
    summary = f"""
    **Analysis Mode:** ⚙️ Standard Algorithm (Rule-Based)
    
    **1. Credit Assessment:**
    The borrower **{company}** falls into the **{bucket}** category (Score: {int(score)}/100).
    
    **2. Forensic Findings:**
    {flag_text}
    
    **3. Recommendation:**
    {action} based on quantitative thresholds.
    """
    return bucket, action, summary, color

def get_gemini_memo(row, company, api_key):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Clean prompt with no backslashes
        prompt = f"""
        Act as a senior credit underwriter. Write a formal credit memo for **{company}** based on:
        - Revenue: {row['Revenue']} | Net Profit: {row['PAT']} | CFO: {row['CFO']}
        - Z-Score: {row['Z_Score']:.2f} | Debt/Equity: {row['Debt_Equity']:.2f}
        
        Task:
        1. Determine Risk Profile.
        2. Identify top strengths and risks.
        3. Comment on Earnings Quality.
        4. Give a final recommendation.
        """
        response = model.generate_content(prompt)
        return "AI ANALYZED", "🤖 SEE MEMO BELOW", f"**Analysis Mode:** 🧠 Live AI (Gemini Pro)\n\n{response.text}", "#4e8cff"
        
    except Exception as e:
        return "ERROR", "API ERROR", f"⚠️ API Failed: {str(e)}\nUsing Rule-Based Logic instead.", "#ff4b4b"

# --- 5. MAIN UI ---
def main():
    st.sidebar.title("🏦 Credit Underwriter")
    
    mode = st.sidebar.radio("Data Source:", ["📂 Select from Dataset", "✍️ Manual Data Entry"])
    
    with st.sidebar.expander("🤖 AI Configuration"):
        api_key = st.text_input("Gemini API Key (Optional)", type="password")

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

    if row is not None:
        st.title(f"🏢 Credit Report: {row['Company']}")
        
        tabs = st.tabs(["1️⃣ Overview", "2️⃣ Financials", "3️⃣ DuPont", "4️⃣ Forensic", "5️⃣ Distress", "6️⃣ Cash Flow", "7️⃣ AI Decision"])
        
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
            c3.metric("Total Debt", f"₹{row['TotalDebt']:,.0f}")
            c4.metric("Risk Score", f"{int(row['Credit_Score'])}", delta_color="off")
            st.markdown("---")
            score = row['Credit_Score']
            b_disp = "LOW RISK" if score > 75 else "MEDIUM" if score > 50 else "HIGH"
            st.info(f"**Calculated Risk Category:** {b_disp}")

        with tabs[1]:
            st.subheader("Liquidity & Profitability")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
            c2.metric("Debt/Equity", f"{row['Debt_Equity']:.2f}x")
            c3.metric("ROE", f"{row['ROE']:.1f}%")

        with tabs[2]:
            st.subheader("DuPont Analysis")
            st.bar_chart(pd.DataFrame({'Val': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']]}, index=['Margin', 'Turnover', 'Leverage', 'ROE']))

        with tabs[3]:
            st.subheader("Forensic Checks")
            c1, c2 = st.columns(2)
            c1.success("✅ Cash Flow OK") if row['CFO'] > row['PAT'] else c1.error("❌ Weak Cash Flow")
            c2.success("✅ Revenue OK") if row['Beneish_Flag_DSRI'] == 0 else c2.error("❌ High Receivables")

        with tabs[4]:
            st.metric("Altman Z-Score", f"{row['Z_Score']:.2f}")
            st.progress(min(max(row['Z_Score']/5, 0), 1))

        with tabs[5]:
            st.info(f"Stage: {row['Life_Cycle']}")
            st.bar_chart(pd.DataFrame({'Flow': [row['CFO'], row['CFI'], row['CFF']]}, index=['Ops', 'Inv', 'Fin']))

        with tabs[6]:
            st.subheader("🤖 Smart Credit Decision")
            if api_key:
                bucket, action, summary, color = get_gemini_memo(row, row['Company'], api_key)
            else:
                bucket, action, summary, color = get_rule_based_memo(row, row['Company'])
            
            st.markdown(f"""<div class="report-box" style="border-left: 5px solid {color};">
                <h2>{action}</h2>
                <pre style="white-space: pre-wrap; font-family: inherit;">{summary}</pre>
            </div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
