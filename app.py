"""
Streamlit UI for Three-Way Matching Invoice System
Main application interface with multi-agent orchestration
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import json
import asyncio
from typing import Dict, Optional
import base64
from io import StringIO

# Import our modules
from agents import InvoiceMatchingOrchestrator, CONTROL_TESTS
from matcher import ThreeWayMatcher
from data_generator import InvoiceDataGenerator

# Page configuration
st.set_page_config(
    page_title="Three-Way Invoice Matching System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
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
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #155724;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        color: #721c24;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        color: #856404;
    }
    .audit-trail {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-family: monospace;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'orchestrator' not in st.session_state:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.session_state.orchestrator = InvoiceMatchingOrchestrator(api_key)
        else:
            st.session_state.orchestrator = None
    
    if 'query_history' not in st.session_state:
        st.session_state.query_history = []
    
    if 'control_test_results' not in st.session_state:
        st.session_state.control_test_results = {}
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False

# Helper functions
def load_sample_data():
    """Load sample data from the data directory"""
    data_dir = "data"
    if os.path.exists(data_dir):
        try:
            invoices_df = pd.read_csv(os.path.join(data_dir, "invoices.csv"))
            pos_df = pd.read_csv(os.path.join(data_dir, "purchase_orders.csv"))
            grs_df = pd.read_csv(os.path.join(data_dir, "goods_receipts.csv"))
            return invoices_df, pos_df, grs_df
        except Exception as e:
            st.error(f"Error loading sample data: {e}")
            return None, None, None
    return None, None, None

def download_button(data, filename, label):
    """Create a download button for data"""
    if isinstance(data, pd.DataFrame):
        csv = data.to_csv(index=False)
    else:
        csv = str(data)
    
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{label}</a>'
    return href

def display_metrics(statistics: Dict):
    """Display key metrics in a nice layout"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Invoices",
            value=statistics.get('total_invoices', 0)
        )
    
    with col2:
        matched = statistics.get('total_matched', 0)
        total = statistics.get('total_invoices', 1)
        match_rate = (matched / total * 100) if total > 0 else 0
        st.metric(
            label="Matched",
            value=matched,
            delta=f"{match_rate:.1f}%"
        )
    
    with col3:
        exceptions = statistics.get('total_exceptions', 0)
        st.metric(
            label="Exceptions",
            value=exceptions,
            delta=f"-{exceptions}" if exceptions > 0 else "0",
            delta_color="inverse"
        )
    
    with col4:
        exposure = statistics.get('total_financial_exposure', 0)
        st.metric(
            label="Financial Exposure",
            value=f"${exposure:,.2f}",
            delta_color="inverse"
        )

def display_audit_trail(audit_trail: list):
    """Display the audit trail in an expandable section"""
    with st.expander("🔍 Audit Trail - Processing Steps", expanded=False):
        for step in audit_trail:
            st.markdown(f"""
            <div class="audit-trail">
                <strong>{step['step']}</strong> - {step['timestamp']}<br>
                {step['details']}
            </div>
            """, unsafe_allow_html=True)

def create_exception_chart(exceptions_df: pd.DataFrame):
    """Create visualization for exceptions"""
    if exceptions_df.empty:
        st.info("No exceptions to visualize")
        return
    
    # Exception type distribution
    exception_counts = exceptions_df['Exception_Type'].value_counts()
    
    fig1 = px.bar(
        x=exception_counts.index,
        y=exception_counts.values,
        labels={'x': 'Exception Type', 'y': 'Count'},
        title='Exceptions by Type',
        color=exception_counts.values,
        color_continuous_scale='Reds'
    )
    
    # Risk level distribution
    if 'Risk_Level' in exceptions_df.columns:
        risk_counts = exceptions_df['Risk_Level'].value_counts()
        fig2 = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title='Risk Level Distribution',
            color_discrete_map={
                'Critical': '#dc3545',
                'High': '#fd7e14',
                'Medium': '#ffc107',
                'Low': '#28a745'
            }
        )
    else:
        fig2 = None
    
    # Financial impact by exception type
    if 'Financial_Impact' in exceptions_df.columns:
        impact_by_type = exceptions_df.groupby('Exception_Type')['Financial_Impact'].sum().sort_values(ascending=True)
        fig3 = px.bar(
            x=impact_by_type.values,
            y=impact_by_type.index,
            orientation='h',
            labels={'x': 'Financial Impact ($)', 'y': 'Exception Type'},
            title='Financial Impact by Exception Type',
            color=impact_by_type.values,
            color_continuous_scale='YlOrRd'
        )
    else:
        fig3 = None
    
    # Display charts
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        if fig2:
            st.plotly_chart(fig2, use_container_width=True)
    
    if fig3:
        st.plotly_chart(fig3, use_container_width=True)

def create_vendor_analysis_chart(vendor_risk_df: pd.DataFrame):
    """Create vendor risk analysis visualization"""
    if vendor_risk_df.empty:
        st.info("No vendor risk data to visualize")
        return
    
    # Top 10 riskiest vendors
    top_vendors = vendor_risk_df.head(10)
    
    fig = px.bar(
        x=top_vendors.index,
        y=top_vendors['Risk_Score'],
        title='Top 10 Riskiest Vendors',
        labels={'x': 'Vendor ID', 'y': 'Risk Score'},
        color=top_vendors['Risk_Score'],
        color_continuous_scale='Reds'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Main Application
def main():
    """Main application logic"""
    init_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Three-Way Invoice Matching System</h1>', unsafe_allow_html=True)
    st.markdown("*Powered by AutoGen Multi-Agent System*")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key configuration
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key for agent processing"
        )
        
        if api_key and not st.session_state.orchestrator:
            st.session_state.orchestrator = InvoiceMatchingOrchestrator(api_key)
            st.success("✓ API Key configured")
        
        st.divider()
        
        # Data Upload Section
        st.header("📁 Data Upload")
        
        upload_option = st.radio(
            "Choose data source:",
            ["Use Sample Data", "Upload CSV Files"]
        )
        
        if upload_option == "Use Sample Data":
            if st.button("Load Sample Data", type="primary"):
                invoices_df, pos_df, grs_df = load_sample_data()
                if invoices_df is not None:
                    st.session_state.invoices_df = invoices_df
                    st.session_state.pos_df = pos_df
                    st.session_state.grs_df = grs_df
                    st.session_state.data_loaded = True
                    st.success("✓ Sample data loaded successfully!")
                else:
                    # Generate sample data if not exists
                    with st.spinner("Generating sample data..."):
                        generator = InvoiceDataGenerator()
                        pos_df, grs_df, invoices_df = generator.generate_all_data()
                        st.session_state.invoices_df = invoices_df
                        st.session_state.pos_df = pos_df
                        st.session_state.grs_df = grs_df
                        st.session_state.data_loaded = True
                        st.success("✓ Sample data generated successfully!")
        else:
            uploaded_invoices = st.file_uploader("Upload Invoices CSV", type=['csv'])
            uploaded_pos = st.file_uploader("Upload Purchase Orders CSV", type=['csv'])
            uploaded_grs = st.file_uploader("Upload Goods Receipts CSV", type=['csv'])
            
            if uploaded_invoices and uploaded_pos and uploaded_grs:
                if st.button("Process Uploaded Files", type="primary"):
                    st.session_state.invoices_df = pd.read_csv(uploaded_invoices)
                    st.session_state.pos_df = pd.read_csv(uploaded_pos)
                    st.session_state.grs_df = pd.read_csv(uploaded_grs)
                    st.session_state.data_loaded = True
                    st.success("✓ Files uploaded successfully!")
        
        # Data Preview
        if st.session_state.data_loaded:
            st.divider()
            st.header("📊 Data Summary")
            st.write(f"**Invoices:** {len(st.session_state.invoices_df)} records")
            st.write(f"**Purchase Orders:** {len(st.session_state.pos_df)} records")
            st.write(f"**Goods Receipts:** {len(st.session_state.grs_df)} records")
        
        st.divider()
        
        # Control Tests Library
        st.header("🧪 Control Test Library")
        selected_test = st.selectbox(
            "Select a control test:",
            options=["Custom Query"] + list(CONTROL_TESTS.keys()),
            help="Choose a predefined control test or create a custom query"
        )
        
        if selected_test != "Custom Query":
            test_info = CONTROL_TESTS[selected_test]
            st.info(f"**Description:** {test_info['description']}")
            st.warning(f"**Severity:** {test_info['severity']}")
    
    # Main Content Area
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load data to begin analysis")
        st.info("Use the sidebar to either load sample data or upload your own CSV files")
        return
    
    if not st.session_state.orchestrator:
        st.error("⚠️ Please configure your OpenAI API key in the sidebar")
        return
    
    # Query Interface
    st.header("🔍 Query Interface")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        if selected_test == "Custom Query":
            query = st.text_input(
                "Enter your query:",
                placeholder="e.g., 'Find all invoices without purchase orders'",
                help="Use natural language to query the data"
            )
        else:
            query = CONTROL_TESTS[selected_test]["query"]
            st.text_input("Query to execute:", value=query, disabled=True)
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        process_button = st.button("🚀 Process Query", type="primary", use_container_width=True)
    
    # Process Query
    if process_button and query:
        with st.spinner("🤖 Agents processing your query..."):
            # Create progress placeholder
            progress_placeholder = st.empty()
            
            # Show agent activity
            with progress_placeholder.container():
                st.info("📋 Data Mapper Agent: Analyzing data schemas...")
                
            results = st.session_state.orchestrator.run_query(
                st.session_state.invoices_df,
                st.session_state.pos_df,
                st.session_state.grs_df,
                query
            )
            
            progress_placeholder.empty()
            
            if results['success']:
                st.session_state.current_results = results
                st.session_state.query_history.append({
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'query': query,
                    'exceptions_found': len(results['exceptions'])
                })
                
                # Save control test results if applicable
                if selected_test != "Custom Query":
                    st.session_state.control_test_results[selected_test] = results
                
                st.success("✅ Query processed successfully!")
            else:
                st.error(f"❌ Error processing query: {results.get('error', 'Unknown error')}")
    
    # Display Results
    if st.session_state.current_results:
        results = st.session_state.current_results
        
        st.divider()
        
        # Metrics Dashboard
        st.header("📈 Results Dashboard")
        display_metrics(results['statistics'])
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Exceptions",
            "📊 Analytics",
            "📝 Audit Report",
            "🔍 Audit Trail",
            "💬 Agent Insights"
        ])
        
        with tab1:
            st.subheader("Exception Details")
            
            if not results['exceptions'].empty:
                # Filter options
                col1, col2, col3 = st.columns(3)
                with col1:
                    exception_types = ['All'] + results['exceptions']['Exception_Type'].unique().tolist()
                    filter_type = st.selectbox("Filter by Type:", exception_types)
                
                with col2:
                    if 'Risk_Level' in results['exceptions'].columns:
                        risk_levels = ['All'] + results['exceptions']['Risk_Level'].unique().tolist()
                        filter_risk = st.selectbox("Filter by Risk:", risk_levels)
                    else:
                        filter_risk = 'All'
                
                # Apply filters
                filtered_df = results['exceptions'].copy()
                if filter_type != 'All':
                    filtered_df = filtered_df[filtered_df['Exception_Type'] == filter_type]
                if filter_risk != 'All' and 'Risk_Level' in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df['Risk_Level'] == filter_risk]
                
                # Display table
                st.dataframe(
                    filtered_df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Exceptions CSV",
                    data=csv,
                    file_name=f"exceptions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No exceptions found! All invoices matched successfully.")
        
        with tab2:
            st.subheader("Visual Analytics")
            
            if not results['exceptions'].empty:
                create_exception_chart(results['exceptions'])
                
                if not results['vendor_risk'].empty:
                    st.divider()
                    st.subheader("Vendor Risk Analysis")
                    create_vendor_analysis_chart(results['vendor_risk'])
            else:
                st.info("No exceptions to analyze")
        
        with tab3:
            st.subheader("Audit Report")
            
            # Display markdown report
            st.markdown(results['report'])
            
            # Download button for report
            st.download_button(
                label="📥 Download Audit Report",
                data=results['report'],
                file_name=f"audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown"
            )
        
        with tab4:
            st.subheader("Processing Audit Trail")
            display_audit_trail(results['audit_trail'])
        
        with tab5:
            st.subheader("Agent Insights")
            
            # Display agent conversations
            with st.expander("🤖 Matching Agent Insights", expanded=True):
                st.markdown(results.get('matching_insights', 'No insights available'))
            
            with st.expander("📊 Analysis Agent Insights", expanded=True):
                st.markdown(results.get('analysis_insights', 'No insights available'))
    
    # Query History
    if st.session_state.query_history:
        st.divider()
        st.header("📜 Query History")
        
        history_df = pd.DataFrame(st.session_state.query_history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()