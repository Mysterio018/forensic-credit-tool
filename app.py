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
    .report-box {
        background-color: #1e1e1e;
        padding: 25px;
        border-radius: 10px;
        border: 1px solid #444;
        font-family: 'Segoe UI', sans-serif;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("financials_master.csv")
        cols = ['Revenue', 'EBITDA', 'EBIT', 'PAT', 'TotalAssets', 'TotalDebt', 
                'Equity', 'CurrentAssets', 'CurrentLiabilities', 'CFO', 'Interest', 
                'CFI', 'CFF']
        for c in cols:
            if c not in df.columns: df[c] = 0
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        return df
    except FileNotFoundError:
        return pd.DataFrame()

# --- 3. ANALYTICAL ENGINE (Reused for Both Modes) ---
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
    
    # Altman Z-Score
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    X5 = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Z_Score'] = 3.25 + (6.56*X1) + (3.26*X2) + (6.72*X3) + (1.05*X4)
    
    # Piotroski F-Score (Simplified)
    df['F1'] = (df['ROA'] > 0).astype(int)
    df['F2'] = (df['CFO'] > 0).astype(int)
    df['F3'] = (df['CFO'] > df['PAT']).astype(int)
    # Note: Shift logic removed for manual single-row entry to avoid errors
    df['F_Score'] = df['F1'] + df['F2'] + df['F3'] 
    
    return df

# --- 4. AI GENERATOR ---
def generate_ai_credit_note(row, company, year):
    risk_score = 0
    flags = []
    
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

    if risk_score >= 4:
        bucket, action, color = "HIGH RISK", "⛔ REJECT / SENIOR REVIEW", "#ff4b4b"
    elif risk_score >= 2:
        bucket, action, color = "MEDIUM RISK", "⚠️ CAUTION / STRICT COVENANTS", "#ffa700"
    else:
        bucket, action, color = "LOW RISK", "✅ APPROVE LOAN", "#00c04b"
        
    return bucket, action, color, flags

# --- 5. MAIN UI LAYOUT ---
def main():
    st.sidebar.title("🔍 AI Forensic Tool")
    
    # --- MODE SELECTION ---
    mode = st.sidebar.radio("Data Source:", ["📂 Select from Dataset", "✍️ Manual Data Entry"])
    
    if mode == "📂 Select from Dataset":
        raw_df = load_data()
        if raw_df.empty:
            st.error("Data missing! Please upload 'financials_master.csv'.")
            st.stop()
        
        company = st.sidebar.selectbox("Select Borrower", raw_df['Company'].unique())
        year = st.sidebar.selectbox("Select FY", sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True))
        row_input = raw_df[(raw_df['Company'] == company) & (raw_df['Year'] == year)].iloc[0]
        
    else:
        # MANUAL ENTRY FORM
        st.sidebar.markdown("---")
        st.sidebar.subheader("Enter Financials (Cr)")
        
        with st.sidebar.form("manual_input"):
            company = st.text_input("Company Name", "New Borrower Ltd")
            year = st.number_input("Financial Year", 2025)
            
            st.markdown("### P&L Statement")
            rev = st.number_input("Revenue", 10000.0)
            ebitda = st.number_input("EBITDA", 2500.0)
            ebit = st.number_input("EBIT", 2000.0)
            pat = st.number_input("Net Profit (PAT)", 1500.0)
            interest = st.number_input("Interest Expense", 500.0)
            
            st.markdown("### Balance Sheet")
            ta = st.number_input("Total Assets", 15000.0)
            debt = st.number_input("Total Debt", 5000.0)
            equity = st.number_input("Total Equity", 8000.0)
            ca = st.number_input("Current Assets", 6000.0)
            cl = st.number_input("Current Liabilities", 4000.0)
            
            st.markdown("### Cash Flow")
            cfo = st.number_input("CFO (Operating)", 1200.0)
            cfi = st.number_input("CFI (Investing)", -500.0)
            cff = st.number_input("CFF (Financing)", -200.0)
            
            submitted = st.form_submit_button("Analyze Manual Data")
            
        if submitted:
            # Create a 1-row DataFrame
            data = {
                'Company': [company], 'Year': [year], 'Revenue': [rev], 'EBITDA': [ebitda],
                'EBIT': [ebit], 'PAT': [pat], 'Interest': [interest], 'TotalAssets': [ta],
                'TotalDebt': [debt], 'Equity': [equity], 'CurrentAssets': [ca],
                'CurrentLiabilities': [cl], 'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff]
            }
            row_input = pd.DataFrame(data).iloc[0]
        else:
            st.info("👈 Enter data in the sidebar and click 'Analyze Manual Data'")
            st.stop()

    # --- PROCESS DATA ---
    # Convert single row to DataFrame for calculation engine
    input_df = pd.DataFrame([row_input])
    df_calc = calculate_metrics(input_df)
    row = df_calc.iloc[0]

    # --- DASHBOARD UI ---
    st.title(f"🏢 {company}: Forensic Credit Report")
    
    # Top Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Revenue", f"₹{row['Revenue']:,.0f}")
    c2.metric("Net Profit", f"₹{row['PAT']:,.0f}")
    c3.metric("Total Debt", f"₹{row['TotalDebt']:,.0f}")
    c4.metric("CFO", f"₹{row['CFO']:,.0f}", delta="Healthy" if row['CFO'] > row['PAT'] else "Weak")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["📊 Ratios", "🔎 Forensic & Fraud", "🧬 DuPont Analysis", "🤖 AI Decision"])
    
    with t1:
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

    with t2:
        st.subheader("Forensic Red Flags")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Altman Z-Score:** {row['Z_Score']:.2f}")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = row['Z_Score'],
                gauge = {'axis': {'range': [None, 5]}, 'bar': {'color': "white"},
                         'steps': [{'range': [0, 1.23], 'color': "#ff4b4b"}, {'range': [1.23, 2.9], 'color': "#ffa700"}, {'range': [2.9, 5], 'color': "#00c04b"}]}
            ))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Earnings Quality (CFO vs PAT)**")
            fig = px.bar(x=['Reported Profit', 'Real Cash Flow'], y=[row['PAT'], row['CFO']], color=['PAT', 'CFO'])
            st.plotly_chart(fig, use_container_width=True)

    with t3:
        st.subheader("ROE Decomposition")
        dupont = pd.DataFrame({
            'Driver': ['Net Margin', 'Asset Turnover', 'Leverage', 'ROE'],
            'Value': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage'], row['ROE']],
            'Type': ['Input', 'Input', 'Input', 'Output']
        })
        fig = px.bar(dupont, x='Driver', y='Value', color='Type', text_auto='.2f', title="ROE Breakdown")
        st.plotly_chart(fig, use_container_width=True)

    with t4:
        bucket, action, color, flags = generate_ai_credit_note(row, company, year)
        st.markdown(f"""
        <div class="report-box" style="border-left: 5px solid {color};">
            <h2>{action}</h2>
            <h4>Risk Profile: <span style="color:{color}">{bucket}</span></h4>
            <hr>
            <p><strong>AI Assessment:</strong> Based on the data provided for {company} (FY{year}), the system calculates a Z-Score of {row['Z_Score']:.2f}.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if flags:
            st.subheader("🚨 Risk Drivers")
            for f in flags: st.error(f"• {f}")
        else:
            st.success("✅ No major forensic flags detected.")

if __name__ == "__main__":
    main()
