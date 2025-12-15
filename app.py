import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import google.generativeai as genai

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI-Based Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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

# --- 2. SECURE AI CONFIGURATION ---
# Try to get the key from Streamlit Secrets (Backend)
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    API_KEY = None  # If not found, AI features will be disabled

# --- 3. DATA LOADING ENGINE ---
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

# --- 4. UNIFIED ANALYSIS ENGINE ---
def calculate_metrics(df):
    if df.empty: return df
    
    # Ratios
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
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
    df['Beneish_Flag'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # Z-Score
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
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

    # Risk Score
    def get_score(row):
        score = 100
        if row['Z_Score'] < 1.23: score -= 25
        elif row['Z_Score'] < 2.9: score -= 10
        if row['CFO_to_PAT'] < 0.8: score -= 15
        if row['Debt_Equity'] > 2.0: score -= 15
        if row['Current_Ratio'] < 1.0: score -= 10
        if row['ICR'] < 1.5: score -= 10
        if row['Beneish_Flag'] == 1: score -= 10
        return max(0, score)
    df['Credit_Score'] = df.apply(get_score, axis=1)
    
    return df

# --- 5. AI GENERATORS ---
def get_rule_based_summary(row, company):
    score = row['Credit_Score']
    if score >= 75: bucket, action, color = "LOW RISK", "✅ APPROVE", "#00c04b"
    elif score >= 50: bucket, action, color = "MEDIUM RISK", "⚠️ CAUTION", "#ffa700"
    else: bucket, action, color = "HIGH RISK", "⛔ REJECT", "#ff4b4b"
    
    flags = []
    if row['Z_Score'] < 1.23: flags.append(f"High Distress Risk (Z: {row['Z_Score']:.2f})")
    if row['CFO_to_PAT'] < 0.8: flags.append("Poor Earnings Quality")
    if row['Beneish_Flag'] == 1: flags.append("Aggressive Revenue Recognition")
    
    flag_text = "\n".join([f"- {f}" for f in flags]) if flags else "- No critical forensic red flags."
    
    return f"""
    **Analysis Mode:** ⚙️ Standard Rules
    
    **1. Risk Assessment**
    Score: {int(score)}/100 ({bucket})
    
    **2. Forensic Findings**
    {flag_text}
    
    **3. Recommendation**
    {action} based on quantitative thresholds.
    """, color

def get_gemini_summary(row, company):
    if not API_KEY:
        return get_rule_based_summary(row, company)

    try:
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        
        prompt = f"""
        Act as a senior credit risk officer. Write a formal credit memo for **{company}**.
        
        **Financial Data:**
        - Revenue: {row['Revenue']} | Net Profit: {row['PAT']} | CFO: {row['CFO']}
        - Debt: {row['TotalDebt']} | Equity: {row['Equity']}
        
        **Forensic Indicators:**
        - Z-Score: {row['Z_Score']:.2f}
        - Earnings Quality (CFO/PAT): {row['CFO_to_PAT']:.2f}
        - Beneish Flag: {'Detected' if row['Beneish_Flag']==1 else 'None'}
        - Life Cycle Stage: {row['Life_Cycle']}
        
        **Task:**
        1. Classify Risk (Low/Medium/High).
        2. Analyze Strengths vs Risks.
        3. Provide a Forensic Verdict.
        4. Final Recommendation (Approve/Reject).
        """
        response = model.generate_content(prompt)
        return f"**Analysis Mode:** 🧠 Live AI (Gemini)\n\n{response.text}", "#4e8cff"
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}\n\n(Falling back to rules...)", "#ff4b4b"

# --- 6. MAIN UI ---
def main():
    st.sidebar.title("🏦 Forensic Credit Tool")
    
    mode = st.sidebar.radio("Input Mode:", ["📂 Dataset Mode", "✍️ Manual Calculator"])
    
    st.sidebar.markdown("---")
    if API_KEY:
        st.sidebar.success("✅ AI Engine Connected")
    else:
        st.sidebar.warning("⚠️ No API Key Found (Using Rules)")

    row = None
    
    if mode == "📂 Dataset Mode":
        raw_df = load_dataset()
        if raw_df.empty:
            st.error("⚠️ Data missing. Upload 'financials_master.csv' to GitHub.")
            st.stop()
        
        company = st.sidebar.selectbox("Borrower", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("FY", years)
        
        if st.sidebar.button("Run Forensic Analysis"):
            df_proc = calculate_metrics(raw_df)
            row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]
            
    else:
        with st.sidebar.form("manual"):
            st.subheader("Financial Input")
            company_input = st.text_input("Company", "New Entity")
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
                    'TotalAssets': [ta], 'Receivables': [rec], 'CFI': [-500], 'CFF': [-200]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    if row is not None:
        st.title(f"🔍 Credit Report: {row['Company']}")
        
        tabs = st.tabs([
            "1️⃣ Overview", "2️⃣ Financials", "3️⃣ DuPont", 
            "4️⃣ Forensic", "5️⃣ Distress", "6️⃣ Cash Flow", "7️⃣ AI Decision"
        ])
        
        # TAB 1: OVERVIEW
        with tabs[0]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
            c3.metric("Debt", f"₹{row['TotalDebt']:,.0f}")
            c4.metric("Risk Score", f"{int(row['Credit_Score'])}", delta_color="off")
            st.markdown("---")
            score = row['Credit_Score']
            bucket = "LOW RISK" if score > 75 else "MEDIUM" if score > 50 else "HIGH"
            st.info(f"**Calculated Risk Bucket:** {bucket}")

        # TAB 2: FINANCIALS
        with tabs[1]:
            st.subheader("Core Ratios")
            c1, c2, c3 = st.columns(3)
            c1.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
            c2.metric("Debt/Equity", f"{row['Debt_Equity']:.2f}x")
            c3.metric("ROE", f"{row['ROE']:.1f}%")

        # TAB 3: DUPONT
        with tabs[2]:
            st.subheader("DuPont Decomposition")
            d_data = pd.DataFrame({
                'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
                'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']]
            })
            st.bar_chart(d_data.set_index('Driver'))
            st.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}")

        # TAB 4: FORENSIC
        with tabs[3]:
            st.subheader("Manipulation Checks")
            c1, c2 = st.columns(2)
            c1.metric("CFO / PAT", f"{row['CFO_to_PAT']:.2f}")
            if row['CFO'] < row['PAT']: c1.error("❌ Weak Cash Conv.")
            else: c1.success("✅ Cash Backed")
            
            c2.metric("Beneish Proxy Flag", "Detected" if row['Beneish_Flag']==1 else "None")
            if row['Beneish_Flag']==1: c2.error("❌ Receivables > Sales Growth")
            else: c2.success("✅ Growth Aligned")

        # TAB 5: DISTRESS
        with tabs[4]:
            st.metric("Altman Z-Score", f"{row['Z_Score']:.2f}")
            st.progress(min(max(row['Z_Score']/5, 0), 1))
            
            st.subheader("Early Warning Signals")
            e1, e2 = st.columns(2)
            e1.metric("Liquidity Trend", "Weak" if row['Current_Ratio']<1 else "Stable")
            e2.metric("Interest Cover", "Stressed" if row['ICR']<1.5 else "Good")

        # TAB 6: CASH FLOW
        with tabs[5]:
            st.bar_chart(pd.DataFrame({'Flow': [row['CFO'], row['CFI'], row['CFF']]}, index=['Ops', 'Inv', 'Fin']))
            st.info(f"📍 **Business Life-Cycle:** {row['Life_Cycle']}")
            st.metric("Debt Service Coverage", f"{row['CF_Debt_Cov']:.2f}")

        # TAB 7: AI DECISION
        with tabs[6]:
            st.subheader("🤖 AI-Assisted Credit Note")
            
            if API_KEY:
                summary, color = get_gemini_summary(row, row['Company'])
            else:
                summary, color = get_rule_based_summary(row, row['Company'])
            
            st.markdown(f"""
            <div class="report-box" style="border-left: 5px solid {color};">
                <pre style="white-space: pre-wrap; font-family: inherit; color: white;">{summary}</pre>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
