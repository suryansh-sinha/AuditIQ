"""
Streamlit UI for Three-Way Matching Invoice System
Redesigned with Default Audit Mode + Optional Query Mode
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import asyncio
from typing import Dict

# Import our modules
from agents import (
    DataMapperAgent, 
    MatchingAgent, 
    AnalysisAgent, 
    ReportAgent,
    QueryAgent,
    MatchingRequest,
    AnalysisRequest,
    ReportRequest,
    QueryRequest
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken

# Page configuration
st.set_page_config(
    page_title="Automated Audit System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .mode-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        margin: 0.5rem;
    }
    .audit-mode {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .query-mode {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 2px solid #17a2b8;
    }
    .progress-step {
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        background-color: #f8f9fa;
    }
    .progress-complete {
        border-left-color: #28a745;
        background-color: #d4edda;
    }
    .progress-active {
        border-left-color: #ffc107;
        background-color: #fff3cd;
        animation: pulse 1.5s infinite;
    }
    .progress-pending {
        border-left-color: #6c757d;
        background-color: #e9ecef;
        opacity: 0.6;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .quick-query-btn {
        background-color: #e9ecef;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 0.75rem;
        margin: 0.25rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-query-btn:hover {
        background-color: #d1ecf1;
        border-color: #17a2b8;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'mode': 'audit',  # 'audit' or 'query'
        'data_loaded': False,
        'audit_status': {
            'mapping_complete': False,
            'matching_complete': False,
            'analysis_complete': False,
            'report_generated': False
        },
        'cached_results': {
            'mappings': None,
            'matching_results': None,
            'analysis_results': None,
            'report': None
        },
        'query_history': [],
        'api_key': os.getenv('GEMINI_API_KEY', ''),
        'model_client': None,
        'current_step': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Helper functions
def load_sample_data():
    """Load sample data from the data directory"""
    data_dir = "data"
    if os.path.exists(data_dir):
        try:
            invoices_df = pd.read_csv(os.path.join(data_dir, "invoices.csv"), keep_default_na=False)
            pos_df = pd.read_csv(os.path.join(data_dir, "purchase_orders.csv"), keep_default_na=False)
            grs_df = pd.read_csv(os.path.join(data_dir, "goods_receipts.csv"), keep_default_na=False)
            return invoices_df, pos_df, grs_df
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return None, None, None
    return None, None, None

def display_progress_tracker():
    """Display audit progress tracker"""
    status = st.session_state.audit_status
    
    steps = [
        ("Schema Analysis", status['mapping_complete'], "📋"),
        ("Three-Way Matching", status['matching_complete'], "🔄"),
        ("Risk Analysis", status['analysis_complete'], "📊"),
        ("Report Generation", status['report_generated'], "📄")
    ]
    
    st.markdown("### 📍 Audit Progress")
    
    for step_name, is_complete, icon in steps:
        if is_complete:
            status_class = "progress-complete"
            status_icon = "✅"
        elif st.session_state.current_step == step_name:
            status_class = "progress-active"
            status_icon = "⏳"
        else:
            status_class = "progress-pending"
            status_icon = "⏸️"
        
        st.markdown(f"""
        <div class="progress-step {status_class}">
            {status_icon} {icon} <strong>{step_name}</strong>
        </div>
        """, unsafe_allow_html=True)

def display_metrics_dashboard(statistics: Dict):
    """Display metrics in card format"""
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        ("Total Invoices", statistics.get('total_invoices', 0), "📝"),
        ("Match Rate", f"{statistics.get('match_rate_pct', 0):.1f}%", "✅"),
        ("Exceptions", statistics.get('total_exceptions', 0), "⚠️"),
        ("Exposure", f"${statistics.get('financial_exposure', 0):,.0f}", "💰")
    ]
    
    for col, (label, value, icon) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 2rem;">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

async def run_audit_workflow():
    """Execute the complete audit workflow"""
    # Create a placeholder for progress
    progress_container = st.empty()
    
    try:
        # Initialize model client
        if not st.session_state.model_client:
            st.session_state.model_client = OpenAIChatCompletionClient(
                model='gemini-2.5-flash',
                api_key=st.session_state.api_key
            )
        
        model_client = st.session_state.model_client
        
        # Get data
        invoices_df = st.session_state.invoices_df
        pos_df = st.session_state.pos_df
        grs_df = st.session_state.grs_df
        
        # Step 1: Schema Mapping
        st.session_state.current_step = "Schema Analysis"
        with progress_container.container():
            with st.spinner("🔍 Analyzing data structure and mapping columns..."):
                display_progress_tracker()
                mapper = DataMapperAgent(api_key=st.session_state.api_key)
                mappings = await mapper.analyze_schemas(invoices_df, pos_df, grs_df)
                st.session_state.cached_results['mappings'] = mappings
                st.session_state.audit_status['mapping_complete'] = True
        
        with progress_container.container():
            st.success(f"✅ Schema mapping completed (Confidence: {mappings.get('confidence', 0):.2%})")
            display_progress_tracker()
        
        # Step 2: Three-Way Matching
        st.session_state.current_step = "Three-Way Matching"
        with progress_container.container():
            with st.spinner("🔄 Matching Invoices ↔ Purchase Orders ↔ Goods Receipts..."):
                display_progress_tracker()
                matching_agent = MatchingAgent(model_client)
                matching_results = await matching_agent.execute_matching(
                    invoices_df, pos_df, grs_df,
                    mappings=mappings,
                    cancellation_token=CancellationToken()
                )
                st.session_state.cached_results['matching_results'] = matching_results
                st.session_state.audit_status['matching_complete'] = True
        
        stats = matching_results['statistics']
        with progress_container.container():
            st.success(f"✅ Matching completed: {stats['total_matched']:,} matched, {stats['total_exceptions']:,} exceptions")
            display_progress_tracker()
        
        # Step 3: Risk Analysis
        st.session_state.current_step = "Risk Analysis"
        with progress_container.container():
            with st.spinner("📊 Analyzing patterns, risks, and fraud indicators..."):
                display_progress_tracker()
                analysis_agent = AnalysisAgent(model_client)
                analysis_results = await analysis_agent.analyze_patterns(
                    exceptions_df=matching_results['exceptions'],
                    matched_df=matching_results['matched'],
                    vendor_risk_df=matching_results.get('vendor_risk'),
                    prepared_inv=matching_results.get('prepared_inv'),
                    cancellation_token=CancellationToken()
                )
                st.session_state.cached_results['analysis_results'] = analysis_results
                st.session_state.audit_status['analysis_complete'] = True
        
        summary = analysis_results['summary']
        with progress_container.container():
            st.success(f"✅ Analysis completed: {len(analysis_results['patterns'])} patterns identified")
            display_progress_tracker()
        
        # Step 4: Report Generation
        st.session_state.current_step = "Report Generation"
        with progress_container.container():
            with st.spinner("📄 Generating comprehensive audit report..."):
                display_progress_tracker()
                report_agent = ReportAgent(model_client)
                
                # Prepare matching results dict
                matching_dict = {
                    'statistics': matching_results['statistics'],
                    'exceptions': matching_results['exceptions'],
                    'matched': matching_results['matched'],
                    'vendor_risk': matching_results.get('vendor_risk'),
                    'prepared_inv': matching_results.get('prepared_inv')
                }
                
                # Prepare analysis results dict
                analysis_dict = {
                    'summary': analysis_results['summary'],
                    'patterns': analysis_results['patterns'],
                    'recommendations': analysis_results['recommendations'],
                    'fraud_indicators': analysis_results.get('fraud_indicators', []),
                    'analysis': analysis_results['analysis']
                }
                
                report_request = ReportRequest(
                    matching_results=matching_dict,
                    analysis_results=analysis_dict,
                    mappings=mappings,
                    company_name="Demo Corporation",
                    auditor_name="AutoGen Audit System"
                )
                
                class DummyContext:
                    cancellation_token = CancellationToken()
                
                report_response = await report_agent.handle_report_request(report_request, DummyContext())
                st.session_state.cached_results['report'] = report_response
                st.session_state.audit_status['report_generated'] = True
        
        st.session_state.current_step = None
        with progress_container.container():
            st.success("✅ Audit report generated successfully!")
            display_progress_tracker()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error during audit: {str(e)}")
        st.session_state.current_step = None
        return False

def render_audit_mode():
    """Render the default audit mode interface"""
    st.markdown('<div class="mode-badge audit-mode">📊 Audit Mode</div>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        # Upload interface
        st.markdown("### 📁 Upload Your Files")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📄 Invoices.csv**")
        with col2:
            st.markdown("**📋 Purchase Orders.csv**")
        with col3:
            st.markdown("**📦 Goods Receipts.csv**")
        
        st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📂 Use Sample Data", type="secondary", width='stretch'):
                invoices_df, pos_df, grs_df = load_sample_data()
                if invoices_df is not None:
                    st.session_state.invoices_df = invoices_df
                    st.session_state.pos_df = pos_df
                    st.session_state.grs_df = grs_df
                    st.session_state.data_loaded = True
                    st.rerun()
        
        with col2:
            st.markdown("**Or upload your own files** ↓")
        
        uploaded_invoices = st.file_uploader("Upload Invoices CSV", type=['csv'], key="inv_upload")
        uploaded_pos = st.file_uploader("Upload Purchase Orders CSV", type=['csv'], key="po_upload")
        uploaded_grs = st.file_uploader("Upload Goods Receipts CSV", type=['csv'], key="gr_upload")
        
        if uploaded_invoices and uploaded_pos and uploaded_grs:
            if st.button("✅ Process Uploaded Files", type="primary", width='stretch'):
                st.session_state.invoices_df = pd.read_csv(uploaded_invoices, keep_default_na=False)
                st.session_state.pos_df = pd.read_csv(uploaded_pos, keep_default_na=False)
                st.session_state.grs_df = pd.read_csv(uploaded_grs, keep_default_na=False)
                st.session_state.data_loaded = True
                st.rerun()
    
    else:
        # Data loaded - show audit interface
        if not st.session_state.audit_status['report_generated']:
            # Pre-audit state
            st.markdown("### ✅ Files Loaded Successfully")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"📝 **Invoices:** {len(st.session_state.invoices_df):,} records")
            with col2:
                st.info(f"📋 **Purchase Orders:** {len(st.session_state.pos_df):,} records")
            with col3:
                st.info(f"📦 **Goods Receipts:** {len(st.session_state.grs_df):,} records")
            
            st.divider()
            
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("▶️ Generate Audit Report", type="primary", width='stretch', key="gen_audit"):
                    asyncio.run(run_audit_workflow())
                    st.rerun()
            with col2:
                if st.button("🔍 Query Mode", width='stretch'):
                    st.session_state.mode = 'query'
                    st.rerun()
        
        else:
            # Post-audit state - show results
            st.markdown("### ✅ Audit Complete!")
            
            # Display metrics
            st.divider()
            results = st.session_state.cached_results
            display_metrics_dashboard(results['matching_results']['statistics'])
            
            st.divider()
            
            # Key findings
            st.markdown("### 📊 Key Findings")
            col1, col2 = st.columns(2)
            
            with col1:
                analysis = results['analysis_results']
                st.markdown("**🎯 Patterns Identified:**")
                for i, pattern in enumerate(analysis['patterns'][:5], 1):
                    st.markdown(f"{i}. {pattern}")
            
            with col2:
                st.markdown("**💡 Recommendations:**")
                for i, rec in enumerate(analysis['recommendations'][:5], 1):
                    st.markdown(f"{i}. {rec}")
            
            if analysis.get('fraud_indicators'):
                st.divider()
                st.error("**🚨 Fraud Indicators Detected:**")
                for indicator in analysis['fraud_indicators']:
                    st.markdown(f"- {indicator}")
            
            st.divider()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                with st.expander("📄 View Full Report", expanded=False):
                    st.markdown(results['report'].markdown_report)
            with col2:
                report_data = results['report'].markdown_report
                st.download_button(
                    label="💾 Download Report",
                    data=report_data,
                    file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    width='stretch'
                )
            with col3:
                if st.button("🔍 Query Data", width='stretch'):
                    st.session_state.mode = 'query'
                    st.rerun()
            
            # Detailed tabs
            st.divider()
            st.markdown("### 📑 Detailed Analysis")
            
            tab1, tab2, tab3 = st.tabs(["📋 Exceptions", "📊 Analytics", "🏢 Vendor Risk"])
            
            with tab1:
                exceptions_df = results['matching_results']['exceptions']
                if not exceptions_df.empty:
                    st.dataframe(exceptions_df, width='stretch', hide_index=True)
                    
                    csv = exceptions_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Exceptions CSV",
                        data=csv,
                        file_name=f"exceptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.success("✅ No exceptions found!")
            
            with tab2:
                if not exceptions_df.empty and 'type' in exceptions_df.columns:
                    exception_counts = exceptions_df['type'].value_counts()
                    fig = px.bar(
                        x=exception_counts.index,
                        y=exception_counts.values,
                        labels={'x': 'Exception Type', 'y': 'Count'},
                        title='Exceptions by Type'
                    )
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No data to visualize")
            
            with tab3:
                vendor_risk = results['analysis_results'].get('vendor_risk')
                if vendor_risk is not None and not vendor_risk.empty:
                    st.dataframe(vendor_risk.head(20), width='stretch')
                else:
                    st.info("No vendor risk data available")
            
            # Reset button
            st.divider()
            if st.button("🔄 Start New Audit", type="secondary"):
                # Reset audit state
                st.session_state.audit_status = {
                    'mapping_complete': False,
                    'matching_complete': False,
                    'analysis_complete': False,
                    'report_generated': False
                }
                st.session_state.cached_results = {
                    'mappings': None,
                    'matching_results': None,
                    'analysis_results': None,
                    'report': None
                }
                st.session_state.data_loaded = False
                st.rerun()

def render_query_mode():
    """Render the query mode interface"""
    st.markdown('<div class="mode-badge query-mode">🔍 Query Mode (Powered by AI)</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Audit", width='stretch'):
            st.session_state.mode = 'audit'
            st.rerun()
    with col2:
        st.markdown("**Ask anything about your audit data...**")
    
    st.divider()
    
    # Query input
    query = st.text_area(
        "Your Question:",
        placeholder="e.g., 'Show me all invoices from vendor V0001'",
        height=100,
        key="query_input"
    )
    
    col1, col2 = st.columns([4, 1])
    with col2:
        execute_query = st.button("🚀 Send", type="primary", width='stretch')
    
    # Quick queries
    st.markdown("### 💡 Quick Queries:")
    quick_queries = [
        "Show me invoices over $15,000",
        "Which vendors have the most purchase orders?",
        "Find all invoices without purchase orders",
        "Show me price variances greater than 10%",
        "List all unique vendors"
    ]
    
    cols = st.columns(3)
    for i, quick_query in enumerate(quick_queries):
        with cols[i % 3]:
            if st.button(quick_query, key=f"quick_{i}", width='stretch'):
                st.session_state.query_input = quick_query
                query = quick_query
                execute_query = True
    
    st.divider()
    
    # Execute query
    if execute_query and query:
        with st.spinner("🤖 Processing your query..."):
            try:
                # Initialize model client if needed
                if not st.session_state.model_client:
                    st.session_state.model_client = OpenAIChatCompletionClient(
                        model='gemini-2.5-flash',
                        api_key=st.session_state.api_key
                    )
                
                # Get mappings if available
                mappings = st.session_state.cached_results.get('mappings')
                
                # Execute query
                query_agent = QueryAgent(st.session_state.model_client)
                result = asyncio.run(query_agent.execute_query(
                    query=query,
                    invoices_df=st.session_state.invoices_df,
                    pos_df=st.session_state.pos_df,
                    grs_df=st.session_state.grs_df,
                    mappings=mappings,
                    cancellation_token=CancellationToken()
                ))
                
                # Add to history
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': query,
                    'row_count': result['row_count']
                })
                
                # Display results
                st.success(f"✅ Query executed successfully!")
                
                st.markdown("### 📊 Results:")
                st.info(f"**Explanation:** {result['explanation']}")
                st.markdown(f"**📈 Found {result['row_count']} results**")
                
                if result['row_count'] > 0:
                    # Show results
                    st.dataframe(result['results'], width='stretch', hide_index=True)
                    
                    # Download button
                    csv = result['results'].to_csv(index=False)
                    st.download_button(
                        label="📥 Export Results",
                        data=csv,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No results found for this query")
                
                # Show generated code
                with st.expander("💻 View Generated Code"):
                    st.code(result['code'], language='python')
                
            except Exception as e:
                st.error(f"❌ Query failed: {str(e)}")
    
    # Query history
    if st.session_state.query_history:
        st.divider()
        st.markdown("### 📜 Query History")
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, width='stretch', hide_index=True)

def main():
    """Main application logic"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Automated Audit System</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by AutoGen Multi-Agent System*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key
        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=st.session_state.api_key,
            help="Enter your Gemini API key"
        )
        
        if api_key != st.session_state.api_key:
            st.session_state.api_key = api_key
            st.session_state.model_client = None  # Reset client
        
        if api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ Please enter API key")
        
        st.divider()
        
        # Mode indicator
        st.header("📍 Current Mode")
        if st.session_state.mode == 'audit':
            st.markdown('<div class="mode-badge audit-mode">📊 Audit Mode</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="mode-badge query-mode">🔍 Query Mode</div>', unsafe_allow_html=True)
        
        # Progress tracker (only in audit mode)
        if st.session_state.mode == 'audit' and st.session_state.data_loaded:
            st.divider()
            display_progress_tracker()
        
        st.divider()
        
        # Data info
        if st.session_state.data_loaded:
            st.header("📊 Data Summary")
            st.write(f"**Invoices:** {len(st.session_state.invoices_df):,}")
            st.write(f"**Purchase Orders:** {len(st.session_state.pos_df):,}")
            st.write(f"**Goods Receipts:** {len(st.session_state.grs_df):,}")
        
        st.divider()
        
        # About
        st.header("ℹ️ About")
        st.markdown("""
        This system performs automated three-way matching of:
        - 📝 Invoices
        - 📋 Purchase Orders  
        - 📦 Goods Receipts
        
        **Features:**
        - 🤖 AI-powered analysis
        - 📊 Risk assessment
        - 🚨 Fraud detection
        - 📄 Comprehensive reports
        """)
    
    # Main content - route based on mode
    if not st.session_state.api_key:
        st.error("⚠️ Please configure your Gemini API key in the sidebar to begin")
        return
    
    if st.session_state.mode == 'audit':
        render_audit_mode()
    else:
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please load data in Audit Mode first")
            if st.button("← Go to Audit Mode"):
                st.session_state.mode = 'audit'
                st.rerun()
        else:
            render_query_mode()

if __name__ == "__main__":
    main()