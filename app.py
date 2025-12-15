import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. APP CONFIGURATION & PROFESSIONAL STYLING ---
st.set_page_config(
    page_title="Credit Underwriter | Internal Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bank-Grade CSS: Dark Mode, Muted Tones, Card Layouts
st.markdown("""
    <style>
    /* Main Background */
    .main { background-color: #0e1117; }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1e2127;
        padding: 15px;
        border-radius: 6px;
        border-left: 4px solid #3b82f6; /* Corporate Blue */
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Headers & Text */
    h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; color: #e0e0e0; letter-spacing: -0.5px; }
    p, label { color: #b0b0b0; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: 1px solid #333; }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        color: #888;
        border-radius: 4px 4px 0 0;
        font-size: 14px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e2127;
        color: #3b82f6;
        border-bottom: 2px solid #3b82f6;
    }

    /* Credit Decision Panel */
    .verdict-box {
        background-color: #161920;
        padding: 20px;
        border: 1px solid #333;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .verdict-header { font-size: 24px; font-weight: bold; margin-bottom: 10px; }
    .verdict-sub { font-size: 16px; color: #888; }
    
    /* Status Colors */
    .status-pass { color: #10b981; font-weight: 600; } /* Muted Green */
    .status-warn { color: #f59e0b; font-weight: 600; } /* Muted Amber */
    .status-fail { color: #ef4444; font-weight: 600; } /* Muted Red */
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA LOADING ENGINE ---
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("financials_master.csv")
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

# --- 3. UNIFIED CALCULATION ENGINE ---
def calculate_metrics(df):
    if df.empty: return df
    
    # --- Tab 2: Financial Analysis ---
    df['Current_Ratio'] = df['CurrentAssets'] / df['CurrentLiabilities'].replace(0, 1)
    df['OCF_Ratio'] = df['CFO'] / df['CurrentLiabilities'].replace(0, 1)
    df['NPM'] = (df['PAT'] / df['Revenue'].replace(0, 1)) * 100
    df['ROA'] = (df['PAT'] / df['TotalAssets'].replace(0, 1)) * 100
    df['ROE'] = (df['PAT'] / df['Equity'].replace(0, 1)) * 100
    df['Debt_Equity'] = df['TotalDebt'] / df['Equity'].replace(0, 1)
    df['ICR'] = df['EBIT'] / df['Interest'].replace(0, 1)
    
    # --- Tab 3: DuPont ---
    df['Dupont_NPM'] = df['PAT'] / df['Revenue'].replace(0, 1)
    df['Asset_Turnover'] = df['Revenue'] / df['TotalAssets'].replace(0, 1)
    df['Fin_Leverage'] = df['TotalAssets'] / df['Equity'].replace(0, 1)
    
    # --- Tab 4: Forensic ---
    df['CFO_to_PAT'] = df['CFO'] / df['PAT'].replace(0, 1)
    df['Accruals_Ratio'] = (df['PAT'] - df['CFO']) / df['TotalAssets'].replace(0, 1)
    df['Sales_Growth'] = df['Revenue'].pct_change().fillna(0)
    df['Rec_Growth'] = df['Receivables'].pct_change().fillna(0)
    df['Beneish_Flag_DSRI'] = (df['Rec_Growth'] > (df['Sales_Growth'] * 1.3)).astype(int) 
    
    # --- Tab 5: Distress ---
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
    
    # --- Tab 6: Life Cycle ---
    df['CF_Debt_Cov'] = df['CFO'] / df['TotalDebt'].replace(0, 1)
    def get_stage(row):
        cfo, cfi, cff = row['CFO'], row['CFI'], row['CFF']
        if cfo < 0 and cfi < 0 and cff > 0: return "Introduction"
        if cfo > 0 and cfi < 0 and cff > 0: return "Growth"
        if cfo > 0 and cfi < 0 and cff < 0: return "Mature"
        if cfo < 0: return "Decline/Stress"
        return "Transition"
    df['Life_Cycle'] = df.apply(get_stage, axis=1)

    # --- Credit Score Calculation ---
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

# --- 4. CREDIT MEMO GENERATOR (Rule-Based) ---
def generate_formal_memo(row):
    score = row['Credit_Score']
    
    # Determine Status
    if score >= 75:
        verdict, risk_profile, color_hex = "APPROVE", "LOW RISK", "#10b981"
        rec_text = "The borrower demonstrates strong financial health with robust liquidity and solvency metrics. Recommended for approval."
    elif score >= 50:
        verdict, risk_profile, color_hex = "REVIEW", "MEDIUM RISK", "#f59e0b"
        rec_text = "The borrower shows moderate risk. Financials are stable but require stricter covenants regarding leverage or working capital."
    else:
        verdict, risk_profile, color_hex = "REJECT", "HIGH RISK", "#ef4444"
        rec_text = "The borrower exhibits signs of significant financial distress. Lending is not recommended without substantial collateral or guarantees."

    # Forensic Observations
    forensic_notes = []
    if row['Z_Score'] < 1.23: forensic_notes.append("Altman Z-Score indicates potential distress zone.")
    if row['CFO_to_PAT'] < 0.8: forensic_notes.append("Earnings Quality Concern: Operating Cash Flow is significantly lower than Net Profit.")
    if row['Beneish_Flag_DSRI'] == 1: forensic_notes.append("Revenue Recognition Alert: Receivables are growing faster than Revenue.")
    if row['Debt_Equity'] > 2.5: forensic_notes.append("Leverage Alert: Debt-to-Equity ratio exceeds conservative thresholds.")
    
    if not forensic_notes:
        forensic_text = "No material forensic red flags detected in the provided financial statements."
    else:
        forensic_text = "\n".join([f"• {note}" for note in forensic_notes])

    return verdict, risk_profile, color_hex, rec_text, forensic_text

# --- 5. MAIN APPLICATION UI ---
def main():
    # --- SIDEBAR: ANALYST CONTROL PANEL ---
    st.sidebar.title("Credit Underwriter")
    
    mode = st.sidebar.radio("Data Source", ["Select from Dataset", "Manual Data Entry"])
    
    row = None
    
    if mode == "Select from Dataset":
        raw_df = load_dataset()
        if raw_df.empty:
            st.sidebar.error("Master Dataset not found.")
            st.stop()
        
        # Sidebar Inputs
        st.sidebar.subheader("Borrower Selection")
        company = st.sidebar.selectbox("Name", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("Financial Year", years)
        
        if st.sidebar.button("Run Credit Analysis", type="primary"):
            df_proc = calculate_metrics(raw_df)
            row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]

    else:
        # Manual Entry
        with st.sidebar.form("manual_entry"):
            st.subheader("Borrower Details")
            company_input = st.text_input("Name", "New Applicant")
            
            with st.expander("Profit & Loss", expanded=True):
                rev = st.number_input("Revenue", 10000.0)
                ebit = st.number_input("EBIT", 2000.0)
                pat = st.number_input("Net Profit (PAT)", 1500.0)
                interest = st.number_input("Interest Expense", 500.0)
            
            with st.expander("Balance Sheet", expanded=False):
                ta = st.number_input("Total Assets", 15000.0)
                debt = st.number_input("Total Debt", 5000.0)
                equity = st.number_input("Equity", 8000.0)
                ca = st.number_input("Current Assets", 6000.0)
                cl = st.number_input("Current Liab.", 4000.0)
                rec = st.number_input("Receivables", 2000.0)
            
            with st.expander("Cash Flow", expanded=False):
                cfo = st.number_input("CFO (Operating)", 1200.0)
                cfi = st.number_input("CFI (Investing)", -500.0)
                cff = st.number_input("CFF (Financing)", -200.0)
                capex = st.number_input("Capex", -300.0)
            
            if st.form_submit_button("Run Credit Analysis"):
                data = {
                    'Company': [company_input], 'Year': [2025], 'Revenue': [rev], 'EBIT': [ebit], 
                    'PAT': [pat], 'Interest': [interest], 'TotalAssets': [ta], 'TotalDebt': [debt], 
                    'Equity': [equity], 'CurrentAssets': [ca], 'CurrentLiabilities': [cl], 
                    'Receivables': [rec], 'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff], 'Capex': [capex]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    # --- MAIN CONTENT AREA ---
    if row is not None:
        # 1. HEADER
        st.markdown(f"## Credit Report — {row['Company']}")
        st.markdown(f"**FY:** {row['Year']} | **Source:** {mode} | **Status:** Preliminary Draft")
        st.markdown("---")

        # 2. TABS
        tabs = st.tabs([
            "Overview", "Financial Analysis", "DuPont & Earnings", 
            "Forensic Checks", "Distress & EWS", "Cash Flow", "Credit Decision"
        ])

        # TAB 1: OVERVIEW
        with tabs[0]:
            st.subheader("Executive Snapshot")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Revenue", f"{row['Revenue']:,.0f}")
            c2.metric("Net Profit", f"{row['PAT']:,.0f}")
            c3.metric("Total Debt", f"{row['TotalDebt']:,.0f}")
            c4.metric("Op. Cash Flow", f"{row['CFO']:,.0f}")
            c5.metric("Composite Score", f"{int(row['Credit_Score'])}")
            
            # Simple Risk Banner
            score = row['Credit_Score']
            bg_color = "#163c2e" if score > 75 else "#3f2c12" if score > 50 else "#3f1212"
            text_color = "#10b981" if score > 75 else "#f59e0b" if score > 50 else "#ef4444"
            cat_text = "LOW RISK" if score > 75 else "MEDIUM RISK" if score > 50 else "HIGH RISK"
            
            st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 6px; border: 1px solid {text_color}; margin-top: 20px;">
                    <h4 style="margin:0; color: {text_color};">Risk Category: {cat_text}</h4>
                    <p style="margin:0; color: #ccc;">Based on a weighted analysis of solvency, liquidity, and forensic indicators.</p>
                </div>
            """, unsafe_allow_html=True)

        # TAB 2: FINANCIAL ANALYSIS
        with tabs[1]:
            st.subheader("A. Liquidity & Solvency")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
                st.caption("Benchmark: > 1.0")
            with c2:
                st.metric("Debt-to-Equity", f"{row['Debt_Equity']:.2f}x")
                st.caption("Benchmark: < 2.0")
            with c3:
                st.metric("Interest Coverage", f"{row['ICR']:.2f}x")
                st.caption("Benchmark: > 1.5")

            # Chart: Liquidity Benchmark
            fig_liq = go.Figure(go.Bar(
                x=['Current Ratio', 'Benchmark'], 
                y=[row['Current_Ratio'], 1.0],
                marker_color=['#3b82f6', '#555']
            ))
            fig_liq.update_layout(title="Liquidity Position vs Benchmark", height=300, margin=dict(t=30,b=0,l=0,r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_liq, use_container_width=True)

            st.divider()
            st.subheader("B. Profitability")
            c4, c5, c6 = st.columns(3)
            c4.metric("Net Margin", f"{row['NPM']:.1f}%")
            c5.metric("ROA", f"{row['ROA']:.1f}%")
            c6.metric("ROE", f"{row['ROE']:.1f}%")

        # TAB 3: DUPONT & QUALITY
        with tabs[2]:
            st.subheader("A. DuPont Decomposition (ROE Drivers)")
            # Visualizing ROE breakdown
            dupont_df = pd.DataFrame({
                'Component': ['Net Margin', 'Asset Turnover', 'Financial Leverage'],
                'Contribution': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage']],
                'Type': ['Efficiency', 'Efficiency', 'Risk']
            })
            fig_dupont = px.bar(dupont_df, x='Component', y='Contribution', color='Type', 
                                title=f"ROE: {row['ROE']:.1f}% Breakdown", text_auto='.2f',
                                color_discrete_map={'Efficiency': '#3b82f6', 'Risk': '#f59e0b'})
            fig_dupont.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_dupont, use_container_width=True)

            st.divider()
            st.subheader("B. Earnings Quality")
            c1, c2 = st.columns(2)
            c1.metric("CFO / PAT Ratio", f"{row['CFO_to_PAT']:.2f}")
            c2.metric("Accruals Ratio", f"{row['Accruals_Ratio']:.2f}")

        # TAB 4: FORENSIC
        with tabs[3]:
            st.subheader("Forensic Diagnostics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 1. Cash Realization Check")
                val = row['CFO_to_PAT']
                if val < 0.8:
                    st.error(f"Alert: Weak Cash Conversion ({val:.2f}). Paper profits may be high.")
                else:
                    st.success(f"Pass: Healthy Cash Conversion ({val:.2f}).")
            
            with col2:
                st.markdown("#### 2. Revenue Manipulation (Beneish Proxy)")
                flag = row['Beneish_Flag_DSRI']
                if flag == 1:
                    st.error("Alert: Receivables growing significantly faster than Revenue (Channel Stuffing risk).")
                else:
                    st.success("Pass: Revenue growth is consistent with Receivables growth.")

        # TAB 5: DISTRESS
        with tabs[4]:
            st.subheader("Financial Distress Model")
            z = row['Z_Score']
            
            st.metric("Altman Z-Score", f"{z:.2f}")
            st.progress(min(max(z/5, 0), 1))
            
            if z > 2.9:
                st.success("Zone: SAFE (> 2.9)")
            elif z > 1.23:
                st.warning("Zone: GREY (1.23 - 2.9) - Monitor Closely")
            else:
                st.error("Zone: DISTRESS (< 1.23) - High Bankruptcy Risk")

            st.divider()
            st.subheader("Early Warning Signals (EWS)")
            
            ews_list = []
            if row['Current_Ratio'] < 1.0: ews_list.append("Liquidity Stress: Current Ratio < 1.0")
            if row['Debt_Equity'] > 2.0: ews_list.append("Leverage Warning: D/E > 2.0")
            if row['ICR'] < 1.5: ews_list.append("Solvency Warning: Interest Coverage < 1.5")
            if row['CFO'] < 0: ews_list.append("Cash Burn: Negative Operating Cash Flow")
            
            if not ews_list:
                st.info("No active Early Warning Signals detected.")
            else:
                for signal in ews_list:
                    st.markdown(f"🔸 {signal}")

        # TAB 6: CASH FLOW
        with tabs[5]:
            st.subheader("Cash Flow Structure")
            cf_df = pd.DataFrame({
                'Activity': ['Operating', 'Investing', 'Financing'],
                'Amount': [row['CFO'], row['CFI'], row['CFF']]
            })
            fig_cf = px.bar(cf_df, x='Activity', y='Amount', color='Amount', 
                            title="Cash Flow Waterfall", color_continuous_scale='Bluered_r')
            fig_cf.update_layout(height=350, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_cf, use_container_width=True)
            
            st.info(f"**Implied Life Cycle Stage:** {row['Life_Cycle']}")

        # TAB 7: DECISION (Redesigned)
        with tabs[6]:
            # Generate the content
            verdict, r_profile, color, rec, forensics = generate_formal_memo(row)
            
            # 1. Verdict Box
            st.markdown(f"""
                <div class="verdict-box" style="border-left: 5px solid {color};">
                    <div class="verdict-header" style="color: {color};">{verdict}</div>
                    <div class="verdict-sub">Risk Profile: <strong>{r_profile}</strong> | Score: {int(row['Credit_Score'])}/100</div>
                </div>
            """, unsafe_allow_html=True)
            
            # 2. Structured Note
            st.subheader("Credit Assessment Note")
            st.markdown(f"""
            **1. Financial Assessment**
            {rec}
            
            **2. Forensic & Compliance Findings**
            {forensics}
            
            **3. Final Recommendation**
            Based on the quantitative models and forensic ratios, the system recommends a decision to **{verdict}**. 
            This recommendation is automated based on the input financial data and should be verified by a senior credit officer.
            """)
            
            st.caption("Generated by Internal Credit Underwriting System v1.0")

    elif mode == "Select from Dataset" and row is None:
        st.info("👈 Select a company from the sidebar to generate a report.")
    elif mode == "Manual Data Entry" and row is None:
        st.info("👈 Fill in the financial details and click 'Run Credit Analysis'.")

if __name__ == "__main__":
    main()
