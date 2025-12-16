import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- 1. APP CONFIGURATION & STYLING ---
st.set_page_config(
    page_title="AI-Assisted Forensic Credit Assessment Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force Theme & Targeted CSS Fixes
st.markdown("""
    <style>
    /* =============================================
       MAIN THEME RESET
       ============================================= */
    /* Main App Background - White */
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    
    /* Sidebar Background - Light Grey */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    
    /* =============================================
       SIDEBAR WIDGET STYLING (Dark Mode Inputs)
       ============================================= */

    /* 1. SIDEBAR LABELS */
    div[data-testid="stSidebar"] label {
        color: #212529 !important; /* Dark grey text for labels outside boxes */
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    /* 2. SELECTBOX (Dropdowns) -> Dark background, White text */
    /* The main box */
    div[data-testid="stSelectbox"] > div > div {
        background-color: #262730 !important; /* Dark background */
        border: 1px solid #555 !important;
        color: #ffffff !important; /* White text */
    }
    /* The text inside the box */
    div[data-testid="stSelectbox"] div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important; 
    }
    /* The arrow icon */
    div[data-testid="stSelectbox"] svg {
        fill: #ffffff !important; 
    }

    /* 3. TEXT INPUT & NUMBER INPUTS (Manual Entry) -> Dark background, White text */
    /* Target the outer container of the input */
    div[data-testid="stTextInput"] > div > div,
    div[data-testid="stNumberInput"] > div > div {
        background-color: #262730 !important; /* Dark background */
        border: 1px solid #555 !important;
        color: #ffffff !important; /* White text */
    }
    /* Target the input text itself */
    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        caret-color: #ffffff !important; /* White cursor */
    }
    /* Fix +/- buttons in NumberInput so they are visible */
    div[data-testid="stNumberInput"] button {
        color: #ffffff !important;
        background-color: transparent !important;
        border: none !important;
    }
    /* Hover effect for +/- buttons */
    div[data-testid="stNumberInput"] button:hover {
        background-color: #3e404a !important;
    }

    /* 4. SIDEBAR BUTTONS (Run Analysis) -> Dark button, White Text */
    div[data-testid="stSidebar"] button {
        background-color: #343a40 !important;
        color: #ffffff !important;
        border: none;
    }
    div[data-testid="stSidebar"] button p {
        color: #ffffff !important;
        font-weight: bold !important;
    }
    div[data-testid="stSidebar"] button:hover {
        background-color: #23272b !important;
    }

    /* =============================================
       DROPDOWN MENU STYLING (The popup list)
       ============================================= */
    /* The list container */
    div[role="listbox"] {
        background-color: #262730 !important; /* Dark background */
        border: 1px solid #555 !important;
    }
    /* The options inside the list */
    div[role="option"] {
        color: #ffffff !important; /* White text */
        background-color: #262730 !important;
    }
    /* Hover and selection state for options */
    div[role="option"]:hover, div[role="option"][aria-selected="true"] {
        background-color: #3e404a !important; /* Lighter dark for hover */
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
    div[data-testid="stMetricLabel"] {
        color: #555 !important;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        color: #000 !important;
        font-weight: 700;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 20px; 
        border-bottom: 2px solid #f0f0f0; 
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        color: #555;
        font-size: 15px;
        font-weight: 600;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: transparent;
        color: #008000;
        border-bottom: 3px solid #008000;
    }

    /* Verdict Box */
    .verdict-box {
        background-color: #f0fdf4; 
        padding: 25px;
        border: 1px solid #bbf7d0;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #1f2937;
    }
    
    /* Global Text Visibility */
    p, h1, h2, h3, h4, h5, li, span, div {
        color: #000000;
    }
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

# --- 3. UNIFIED CALCULATION ENGINE ---
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
    
    # Distress (Altman Z''-Score for Non-Manufacturing/General Use)
    X1 = (df['CurrentAssets'] - df['CurrentLiabilities']) / df['TotalAssets'].replace(0, 1)
    X2 = df['PAT'] / df['TotalAssets'].replace(0, 1)
    X3 = df['EBIT'] / df['TotalAssets'].replace(0, 1)
    X4 = df['Equity'] / df['TotalDebt'].replace(0, 1)
    # X5 (Sales/TA) is not used in the Z''-score model
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

# --- 4. CREDIT MEMO GENERATOR ---
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

# --- 5. MAIN UI ---
def main():
    st.sidebar.title("AI-Assisted Forensic Credit Assessment Tool")
    mode = st.sidebar.radio("Data Source", ["Select from Dataset", "Manual Data Entry"])
    
    row = None
    
    if mode == "Select from Dataset":
        raw_df = load_dataset()
        if raw_df.empty:
            st.sidebar.error("Master Dataset not found.")
            st.stop()
        
        company = st.sidebar.selectbox("Name", raw_df['Company'].unique())
        years = sorted(raw_df[raw_df['Company'] == company]['Year'].unique(), reverse=True)
        year = st.sidebar.selectbox("Financial Year", years)
        
        if st.sidebar.button("Run Analysis"):
            df_proc = calculate_metrics(raw_df)
            row = df_proc[(df_proc['Company'] == company) & (df_proc['Year'] == year)].iloc[0]

    else:
        with st.sidebar.form("manual_entry"):
            st.subheader("Borrower Details")
            company_input = st.text_input("Name", "New Applicant")
            
            # --- NUMBER INPUTS: ALLOW NEGATIVES & VISIBLE BOXES ---
            with st.expander("Profit & Loss", expanded=True):
                rev = st.number_input("Revenue", value=10000.0, step=100.0, min_value=-1e9, format="%.2f")
                ebit = st.number_input("EBIT", value=2000.0, step=100.0, min_value=-1e9, format="%.2f")
                pat = st.number_input("Net Profit (PAT)", value=1500.0, step=100.0, min_value=-1e9, format="%.2f")
                interest = st.number_input("Interest Expense", value=500.0, step=50.0, min_value=-1e9, format="%.2f")
            
            with st.expander("Balance Sheet", expanded=False):
                ta = st.number_input("Total Assets", value=15000.0, step=100.0, min_value=-1e9, format="%.2f")
                debt = st.number_input("Total Debt", value=5000.0, step=100.0, min_value=-1e9, format="%.2f")
                equity = st.number_input("Equity", value=8000.0, step=100.0, min_value=-1e9, format="%.2f")
                ca = st.number_input("Current Assets", value=6000.0, step=100.0, min_value=-1e9, format="%.2f")
                cl = st.number_input("Current Liab.", value=4000.0, step=100.0, min_value=-1e9, format="%.2f")
                rec = st.number_input("Receivables", value=2000.0, step=100.0, min_value=-1e9, format="%.2f")
            
            with st.expander("Cash Flow", expanded=False):
                cfo = st.number_input("CFO (Operating)", value=1200.0, step=100.0, min_value=-1e9, format="%.2f")
                cfi = st.number_input("CFI (Investing)", value=-500.0, step=100.0, min_value=-1e9, format="%.2f")
                cff = st.number_input("CFF (Financing)", value=-200.0, step=100.0, min_value=-1e9, format="%.2f")
                capex = st.number_input("Capex", value=-300.0, step=100.0, min_value=-1e9, format="%.2f")
            
            if st.form_submit_button("Run Analysis"):
                data = {
                    'Company': [company_input], 'Year': [2025], 'Revenue': [rev], 'EBIT': [ebit], 
                    'PAT': [pat], 'Interest': [interest], 'TotalAssets': [ta], 'TotalDebt': [debt], 
                    'Equity': [equity], 'CurrentAssets': [ca], 'CurrentLiabilities': [cl], 
                    'Receivables': [rec], 'CFO': [cfo], 'CFI': [cfi], 'CFF': [cff], 'Capex': [capex]
                }
                df_proc = calculate_metrics(pd.DataFrame(data))
                row = df_proc.iloc[0]

    if row is not None:
        # HEADER
        st.markdown(f"## Credit Report — {row['Company']}")
        st.markdown(f"**FY:** {row['Year']} | **Source:** {mode} | **Generated by:** Auto-Assessment")
        st.caption("ℹ️ Note: All financial values are in INR Crores.")
        st.markdown("---")

        # TABS
        tabs = st.tabs([
            "Overview", "Financial Analysis", "DuPont & Earnings", 
            "Forensic Checks", "Distress & EWS", "Cash Flow", "Verdict"
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
            
            # --- OVERVIEW CHARTS ---
            st.markdown("### Key Financial Visuals")
            g1, g2 = st.columns(2)
            
            with g1:
                # Capital Structure (Donut)
                cap_fig = go.Figure(data=[go.Pie(
                    labels=['Total Debt', 'Equity'],
                    values=[row['TotalDebt'], row['Equity']],
                    hole=.4,
                    marker_colors=['#dc2626', '#008000'],
                    textfont=dict(color='black')
                )])
                cap_fig.update_layout(
                    title={'text': "Capital Structure (Debt vs Equity)", 'font': {'color': 'black'}},
                    font=dict(color='black'),
                    height=300, 
                    template="plotly_white", 
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(font=dict(color='black'))
                )
                st.plotly_chart(cap_fig, use_container_width=True)
            
            with g2:
                # Profitability Bar
                prof_fig = go.Figure([go.Bar(
                    x=['Revenue', 'EBIT', 'Net Profit'],
                    y=[row['Revenue'], row['EBIT'], row['PAT']],
                    marker_color=['#2563eb', '#3b82f6', '#008000'],
                    texttemplate='%{y:.2s}', textposition='auto',
                    textfont=dict(color='black')
                )])
                prof_fig.update_layout(
                    title={'text': "Profitability Composition", 'font': {'color': 'black'}},
                    font=dict(color='black'),
                    xaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
                    yaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
                    height=300, 
                    template="plotly_white",
                    paper_bgcolor='rgba(0,0,0,0)', 
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(prof_fig, use_container_width=True)

        # TAB 2: FINANCIAL ANALYSIS
        with tabs[1]:
            st.subheader("A. Return Ratios")
            c1, c2, c3 = st.columns(3)
            c1.metric("ROCE %", f"{row['ROCE']:.1f}%")
            c2.metric("ROE %", f"{row['ROE']:.1f}%")
            c3.metric("ROA %", f"{row['ROA']:.1f}%")

            st.divider()
            st.subheader("B. Liquidity & Efficiency")
            c4, c5, c6 = st.columns(3)
            c4.metric("Current Ratio", f"{row['Current_Ratio']:.2f}x")
            c5.metric("Debtor Days", f"{row['Debtor_Days']:.0f} Days")
            c6.metric("Asset Turnover", f"{row['Asset_Turnover']:.2f}x")

            st.divider()
            st.subheader("C. Solvency")
            c7, c8 = st.columns(2)
            c7.metric("Debt-to-Equity", f"{row['Debt_Equity']:.2f}x")
            c8.metric("Interest Coverage", f"{row['ICR']:.2f}x")

        # TAB 3: DUPONT
        with tabs[2]:
            st.subheader("A. DuPont Decomposition")
            dupont_df = pd.DataFrame({
                'Component': ['Net Margin', 'Asset Turnover', 'Financial Leverage'],
                'Contribution': [row['Dupont_NPM']*100, row['Asset_Turnover'], row['Fin_Leverage']],
                'Type': ['Efficiency', 'Efficiency', 'Risk']
            })
            fig_dupont = px.bar(dupont_df, x='Component', y='Contribution', color='Type', 
                                title=f"ROE: {row['ROE']:.1f}% Breakdown", text_auto='.2f',
                                color_discrete_map={'Efficiency': '#3b82f6', 'Risk': '#f59e0b'})
            fig_dupont.update_layout(
                title_font_color="black",
                height=350, 
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='black'),
                xaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
                yaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
                legend=dict(font=dict(color='black'))
            )
            fig_dupont.update_traces(textfont=dict(color='black'))
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
                if val < 0.8: st.error(f"Alert: Weak Cash Conversion ({val:.2f}). Paper profits.")
                else: st.success(f"Pass: Healthy Cash Conversion ({val:.2f}).")
            with col2:
                st.markdown("#### 2. Revenue Manipulation (Beneish Proxy)")
                if row['Beneish_Flag_DSRI'] == 1: st.error("Alert: Receivables growing faster than Revenue.")
                else: st.success("Pass: Revenue growth consistent with Receivables.")

        # TAB 5: DISTRESS
        with tabs[4]:
            st.subheader("Financial Distress Model")
            z = row['Z_Score']
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number", value = z,
                title = {'text': "Altman Z''-Score", 'font': {'color': 'black'}},
                number = {'font': {'color': 'black'}},
                gauge = {
                    'axis': {'range': [None, 5], 'tickfont': {'color': 'black'}},
                    'bar': {'color': "black"},
                    'steps': [{'range': [0, 1.23], 'color': "#ffcccb"}, 
                              {'range': [1.23, 2.9], 'color': "#fff4cc"}, 
                              {'range': [2.9, 5], 'color': "#d4edda"}]
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(t=50, b=50, l=30, r=30), 
                                    template="plotly_white",
                                    paper_bgcolor='rgba(0,0,0,0)', font={'color': 'black'})
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if z > 2.9: st.success("Zone: SAFE (> 2.9)")
            elif z > 1.23: st.warning("Zone: GREY (1.23 - 2.9)")
            else: st.error("Zone: DISTRESS (< 1.23)")

            st.divider()
            st.subheader("Early Warning Signals")
            ews_list = []
            if row['Current_Ratio'] < 1.0: ews_list.append("Liquidity Stress: Current Ratio < 1.0")
            if row['Debt_Equity'] > 2.0: ews_list.append("Leverage Warning: D/E > 2.0")
            if row['ICR'] < 1.5: ews_list.append("Solvency Warning: Interest Coverage < 1.5")
            
            if not ews_list: st.info("No active Early Warning Signals detected.")
            else: 
                for signal in ews_list: st.markdown(f"- {signal}")

        # TAB 6: CASH FLOW
        with tabs[5]:
            st.subheader("Cash Flow Structure")
            cf_df = pd.DataFrame({
                'Activity': ['Operating', 'Investing', 'Financing'],
                'Amount': [row['CFO'], row['CFI'], row['CFF']]
            })
            fig_cf = px.bar(cf_df, x='Activity', y='Amount', color='Amount', 
                            title="Cash Flow Waterfall", color_continuous_scale='Bluered_r',
                            text_auto='.2s')
            
            fig_cf.update_layout(
                title_font_color="black",
                height=350, 
                template="plotly_white",
                paper_bgcolor='rgba(0,0,0,0)', 
                plot_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='black'),
                xaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black')),
                yaxis=dict(tickfont=dict(color='black'), title_font=dict(color='black'))
            )
            fig_cf.update_traces(textfont=dict(color='black'))
            st.plotly_chart(fig_cf, use_container_width=True)
            st.info(f"**Implied Life Cycle Stage:** {row['Life_Cycle']}")

        # TAB 7: DECISION
        with tabs[6]:
            verdict, r_profile, color, rec, forensics = generate_formal_memo(row)
            
            st.markdown(f"""
                <div class="verdict-box">
                    <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 10px;">{verdict}</div>
                    <div style="font-size: 16px; color: #333;">Risk Profile: <strong>{r_profile}</strong> | Score: {int(row['Credit_Score'])}/100</div>
                </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Credit Assessment Note")
            st.markdown(f"""
            <div style="color: #333;">
            <strong>1. Financial Assessment</strong><br>
            {rec}<br><br>
            <strong>2. Forensic & Compliance Findings</strong><br>
            {forensics}<br><br>
            <strong>3. Final Recommendation</strong><br>
            Based on the quantitative models, the system recommends: <strong>{verdict}</strong>.
            </div>
            """, unsafe_allow_html=True)

    elif mode == "Select from Dataset" and row is None:
        st.info("👈 Select a company from the sidebar to generate a report.")
    elif mode == "Manual Data Entry" and row is None:
        st.info("👈 Fill in the financial details and click 'Run Analysis'.")

if __name__ == "__main__":
    main()
