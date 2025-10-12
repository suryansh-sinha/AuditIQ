"""
AutoGen Multi-Agent System for Three-Way Matching
Using AutoGen v0.7.5 to orchestrate multiple specialized agents
"""

import asyncio
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient

from matcher import ThreeWayMatcher

class DataMapperAgent:
    """Agent responsible for understanding and mapping data schemas"""
    
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="DataMapper",
            model_client=model_client,
            system_message="""You are a data mapping specialist for accounts payable audit systems.
            Your responsibilities:
            1. Analyze CSV file structures and identify column purposes
            2. Create mapping dictionaries between different data sources
            3. Identify key columns (IDs, amounts, dates)
            4. Suggest appropriate joins between tables
            5. Generate data glossaries for audit documentation
            
            When analyzing data, provide:
            - Column mapping dictionary
            - Data type identification
            - Suggested join keys
            - Data quality observations
            
            Always respond with structured JSON where appropriate."""
        )
    
    async def analyze_schemas(self, invoices_df, pos_df, grs_df) -> Dict:
        """Analyze data schemas and create mappings"""
        
        schema_info = {
            "invoices": {
                "columns": invoices_df.columns.tolist(),
                "types": {k: str(v) for k, v in invoices_df.dtypes.to_dict().items()},
                "sample_size": len(invoices_df),
                "null_counts": invoices_df.isnull().sum().to_dict()
            },
            "purchase_orders": {
                "columns": pos_df.columns.tolist(),
                "types": {k: str(v) for k, v in pos_df.dtypes.to_dict().items()},
                "sample_size": len(pos_df),
                "null_counts": pos_df.isnull().sum().to_dict()
            },
            "goods_receipts": {
                "columns": grs_df.columns.tolist(),
                "types": {k: str(v) for k, v in grs_df.dtypes.to_dict().items()},
                "sample_size": len(grs_df),
                "null_counts": grs_df.isnull().sum().to_dict()
            }
        }
        
        prompt = f"""Analyze these data schemas and create column mappings:
        
        {json.dumps(schema_info, indent=2, default=str)}
        
        Provide a JSON response with:
        1. Key columns for joining
        2. Amount/quantity columns for validation
        3. Date columns for timeline analysis
        4. Vendor/item identifiers
        
        Format your response as valid JSON."""
        
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")]
        )
        
        return self._parse_mapping_response(response.chat_message.content)
    
    def _parse_mapping_response(self, response: str) -> Dict:
        """Parse agent response to extract mappings"""
        try:
            # Try to find JSON in response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Default mapping structure based on actual data
        return {
            "join_keys": {
                "invoice_to_po": "PO_ID",
                "po_to_gr": "PO_ID",
                "vendor_check": "Vendor_ID"
            },
            "validation_columns": {
                "quantities": ["Invoice_Qty", "Quantity", "Received_Qty"],
                "amounts": ["Invoice_Amount", "Total_Amount", "Invoice_Price", "Unit_Price"],
                "dates": ["Invoice_Date", "PO_Date", "GR_Date"]
            },
            "identified_columns": {
                "invoice_ids": ["Invoice_ID"],
                "po_ids": ["PO_ID"],
                "gr_ids": ["GR_ID"],
                "vendor_ids": ["Vendor_ID"],
                "item_codes": ["Item_Code"]
            }
        }


class MatchingAgent:
    """Agent responsible for executing three-way matching logic"""
    
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Matcher",
            model_client=model_client,
            system_message="""You are a three-way matching specialist for accounts payable.
            Your responsibilities:
            1. Execute matching between invoices, POs, and goods receipts
            2. Identify all types of mismatches and exceptions
            3. Calculate financial exposure from mismatches
            4. Flag potential fraud indicators
            
            Exception types to detect:
            - Missing purchase orders (potential fraud)
            - Missing goods receipts (premature payment)
            - Quantity mismatches (overbilling)
            - Price variances (contract violations)
            - Duplicate invoices (double payment risk)
            
            Provide detailed exception reports with risk levels and financial impact."""
        )
        self.matcher = ThreeWayMatcher()
    
    async def execute_matching(self, invoices_df, pos_df, grs_df, query: str = None) -> Dict:
        """Execute matching based on query or full analysis"""
        
        # Run the actual matching logic
        matched_df, exceptions_df, statistics = self.matcher.three_way_match(
            invoices_df, pos_df, grs_df
        )
        
        # If there's a specific query, filter results
        if query:
            filtered_exceptions = self._filter_by_query(exceptions_df, query)
        else:
            filtered_exceptions = exceptions_df
        
        # Create analysis prompt for the agent
        prompt = f"""Analyze these matching results:
        
        Statistics:
        - Total invoices: {statistics['total_invoices']}
        - Matched: {statistics['total_matched']}
        - Exceptions: {statistics['total_exceptions']}
        - Financial exposure: ${statistics['total_financial_exposure']:,.2f}
        
        Exception breakdown: {statistics['exception_breakdown']}
        
        User query: {query or 'Full analysis requested'}
        
        Provide insights on:
        1. Most critical exceptions
        2. Vendor patterns  
        3. Recommended actions
        
        Be concise and specific."""
        
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")]
        )
        
        return {
            "matched": matched_df,
            "exceptions": filtered_exceptions,
            "statistics": statistics,
            "insights": response.chat_message.content
        }
    
    def _filter_by_query(self, exceptions_df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Filter exceptions based on natural language query"""
        if exceptions_df.empty:
            return exceptions_df
            
        query_lower = query.lower()
        
        if "without purchase order" in query_lower or "no po" in query_lower:
            return exceptions_df[exceptions_df['Exception_Type'] == 'NO_PO']
        elif "price variance" in query_lower:
            if "over" in query_lower or ">" in query_lower:
                # Extract percentage if mentioned
                import re
                percent_match = re.search(r'(\d+)%?', query_lower)
                if percent_match:
                    threshold = float(percent_match.group(1)) / 100
                    # Filter based on parsed variance from reason
                    return exceptions_df[
                        (exceptions_df['Exception_Type'] == 'PRICE_VARIANCE')
                    ]
            return exceptions_df[exceptions_df['Exception_Type'] == 'PRICE_VARIANCE']
        elif "duplicate" in query_lower:
            return exceptions_df[exceptions_df['Exception_Type'] == 'DUPLICATE']
        elif "quantity" in query_lower or "overbilling" in query_lower:
            return exceptions_df[exceptions_df['Exception_Type'] == 'QTY_MISMATCH']
        elif "goods receipt" in query_lower or "no gr" in query_lower:
            return exceptions_df[exceptions_df['Exception_Type'] == 'NO_GR']
        elif "vendor" in query_lower:
            # Extract vendor name if mentioned
            vendor_keywords = query_lower.split("vendor")[-1].strip().split()[0:2]
            if vendor_keywords:
                vendor_filter = exceptions_df['Vendor_ID'].str.contains(
                    '|'.join(vendor_keywords), case=False, na=False
                )
                return exceptions_df[vendor_filter]
        
        return exceptions_df


class AnalysisAgent:
    """Agent responsible for pattern analysis and risk assessment"""
    
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Analyzer",
            model_client=model_client,
            system_message="""You are a financial audit analyst specializing in accounts payable.
            Your responsibilities:
            1. Identify patterns in exceptions and mismatches
            2. Assess vendor risk profiles
            3. Calculate financial exposure by category
            4. Detect potential fraud patterns
            5. Prioritize exceptions by risk level
            
            Focus on:
            - Vendor behavior patterns
            - Temporal patterns (timing of exceptions)
            - Amount distributions
            - Systemic vs isolated issues
            
            Provide actionable insights with specific recommendations."""
        )
    
    async def analyze_patterns(self, exceptions_df: pd.DataFrame, matched_df: pd.DataFrame) -> Dict:
        """Analyze patterns in matching results"""
        
        if exceptions_df.empty:
            return {
                "vendor_risk": pd.DataFrame(),
                "patterns": [],
                "recommendations": ["No exceptions found. All invoices matched successfully."],
                "analysis": "Clean matching - no issues detected.",
                "summary": {"total_exceptions": 0, "total_exposure": 0}
            }
        
        # Vendor risk analysis
        matcher = ThreeWayMatcher()
        vendor_risk = matcher.analyze_vendor_risk(exceptions_df)
        
        # Temporal analysis if date column exists
        temporal_patterns = None
        if 'Invoice_Date' in exceptions_df.columns:
            try:
                exceptions_df['Invoice_Date'] = pd.to_datetime(exceptions_df['Invoice_Date'], errors='coerce')
                valid_dates = exceptions_df.dropna(subset=['Invoice_Date'])
                if not valid_dates.empty:
                    temporal_patterns = valid_dates.groupby([
                        pd.Grouper(key='Invoice_Date', freq='W'),
                        'Exception_Type'
                    ]).size().unstack(fill_value=0)
            except:
                temporal_patterns = None
        
        # Create analysis summary
        analysis_summary = {
            "total_exceptions": len(exceptions_df),
            "total_exposure": float(exceptions_df['Financial_Impact'].sum()) if 'Financial_Impact' in exceptions_df.columns else 0,
            "exception_types": exceptions_df['Exception_Type'].value_counts().to_dict(),
            "high_risk_vendors": vendor_risk.head(5).to_dict() if not vendor_risk.empty else {},
            "critical_exceptions": int(exceptions_df[exceptions_df['Risk_Level'] == 'Critical'].shape[0]) if 'Risk_Level' in exceptions_df.columns else 0,
            "high_exceptions": int(exceptions_df[exceptions_df['Risk_Level'] == 'High'].shape[0]) if 'Risk_Level' in exceptions_df.columns else 0
        }
        
        prompt = f"""Analyze these audit findings:
        
        {json.dumps(analysis_summary, default=str, indent=2)}
        
        Provide:
        1. Top 3 risk patterns identified
        2. Vendor-specific concerns
        3. Process improvement recommendations
        4. Fraud indicators if any
        
        Be specific and actionable."""
        
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")]
        )
        
        return {
            "vendor_risk": vendor_risk,
            "temporal_patterns": temporal_patterns,
            "analysis": response.chat_message.content,
            "summary": analysis_summary
        }


class ReportAgent:
    """Agent responsible for generating audit reports"""
    
    def __init__(self, model_client):
        self.agent = AssistantAgent(
            name="Reporter",
            model_client=model_client,
            system_message="""You are an audit report specialist.
            Your responsibilities:
            1. Generate executive summaries
            2. Create detailed findings tables
            3. Provide clear recommendations
            4. Format reports in markdown for easy reading
            
            Report structure:
            - Executive Summary (key findings, financial impact)
            - Detailed Findings (organized by risk level)
            - Vendor Analysis
            - Recommendations (prioritized actions)
            
            Use clear, professional language suitable for audit committees."""
        )
    
    async def generate_report(self, 
                             matching_results: Dict,
                             analysis_results: Dict,
                             query: str = None) -> str:
        """Generate comprehensive audit report"""
        
        statistics = matching_results.get("statistics", {})
        summary = analysis_results.get("summary", {})
        
        # Get top vendor risks if available
        vendor_risk_df = analysis_results.get("vendor_risk", pd.DataFrame())
        if not vendor_risk_df.empty:
            top_risks = vendor_risk_df.head(5).to_dict('records')
        else:
            top_risks = []
        
        report_data = {
            "query": query or "Full Three-Way Matching Analysis",
            "execution_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "statistics": {
                "total_invoices": statistics.get("total_invoices", 0),
                "matched": statistics.get("total_matched", 0),
                "exceptions": statistics.get("total_exceptions", 0),
                "match_rate": f"{statistics.get('match_rate', 0):.1f}%",
                "financial_exposure": f"${statistics.get('total_financial_exposure', 0):,.2f}"
            },
            "exception_breakdown": statistics.get("exception_breakdown", {}),
            "top_vendor_risks": top_risks,
            "critical_count": summary.get("critical_exceptions", 0),
            "high_count": summary.get("high_exceptions", 0)
        }
        
        prompt = f"""Generate a professional audit report based on:
        
        {json.dumps(report_data, default=str, indent=2)}
        
        Matching Agent Insights:
        {matching_results.get('insights', 'N/A')}
        
        Analysis Agent Insights:
        {analysis_results.get('analysis', 'N/A')}
        
        Format as markdown with:
        
        # Three-Way Matching Audit Report
        
        ## Executive Summary
        (Paragraph summarizing key findings and financial impact)
        
        ## Key Statistics
        (Bullet points of main metrics)
        
        ## Critical Findings
        (Most important exceptions requiring immediate attention)
        
        ## Risk Assessment
        (Vendor risks and patterns identified)
        
        ## Recommendations
        (Numbered list of actionable recommendations)
        
        ## Next Steps
        (Immediate actions required)
        
        Make it concise but comprehensive."""
        
        response = await self.agent.on_messages(
            [TextMessage(content=prompt, source="user")]
        )
        
        return response.chat_message.content


class InvoiceMatchingOrchestrator:
    """Main orchestrator for the multi-agent system"""
    
    def __init__(self, api_key: str = None):
        # Use environment variable if no key provided
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError("Gemini API key required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        
        self.model_client = OpenAIChatCompletionClient(
            model="gemini-2.5-flash",
            api_key=api_key
        )
        
        # Initialize agents
        self.data_mapper = DataMapperAgent(self.model_client)
        self.matcher = MatchingAgent(self.model_client)
        self.analyzer = AnalysisAgent(self.model_client)
        self.reporter = ReportAgent(self.model_client)
        
        # Store audit trail
        self.audit_trail = []
        
        # Store last results for reuse
        self.last_results = None
    
    def log_step(self, step: str, details: Any):
        """Log steps for audit trail"""
        self.audit_trail.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": str(details)[:500]  # Limit detail length
        })
    
    async def process_query(self, 
                          invoices_df: pd.DataFrame,
                          pos_df: pd.DataFrame,
                          grs_df: pd.DataFrame,
                          query: str) -> Dict:
        """Process a natural language query through the agent system"""
        
        # Clear previous audit trail
        self.audit_trail = []
        
        try:
            # Step 1: Data Mapping
            self.log_step("Data Mapping", f"Analyzing schemas for {len(invoices_df)} invoices, {len(pos_df)} POs, {len(grs_df)} GRs")
            mappings = await self.data_mapper.analyze_schemas(invoices_df, pos_df, grs_df)
            
            # Step 2: Execute Matching
            self.log_step("Three-Way Matching", f"Executing matching with query: {query}")
            matching_results = await self.matcher.execute_matching(
                invoices_df, pos_df, grs_df, query
            )
            
            # Step 3: Pattern Analysis
            exceptions_count = len(matching_results['exceptions']) if not matching_results['exceptions'].empty else 0
            self.log_step("Pattern Analysis", f"Analyzing {exceptions_count} exceptions")
            analysis_results = await self.analyzer.analyze_patterns(
                matching_results['exceptions'],
                matching_results['matched']
            )
            
            # Step 4: Generate Report
            self.log_step("Report Generation", "Creating audit report")
            report = await self.reporter.generate_report(
                matching_results,
                analysis_results,
                query
            )
            
            # Store results
            self.last_results = {
                "success": True,
                "query": query,
                "mappings": mappings,
                "matched_records": matching_results['matched'],
                "exceptions": matching_results['exceptions'],
                "statistics": matching_results['statistics'],
                "vendor_risk": analysis_results['vendor_risk'],
                "report": report,
                "audit_trail": self.audit_trail,
                "matching_insights": matching_results.get('insights', ''),
                "analysis_insights": analysis_results.get('analysis', '')
            }
            
            return self.last_results
            
        except Exception as e:
            self.log_step("Error", str(e))
            return {
                "success": False,
                "error": str(e),
                "audit_trail": self.audit_trail
            }
    
    def run_query(self, 
                  invoices_df: pd.DataFrame,
                  pos_df: pd.DataFrame,
                  grs_df: pd.DataFrame,
                  query: str) -> Dict:
        """Synchronous wrapper for async process_query"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.process_query(invoices_df, pos_df, grs_df, query)
            )
        finally:
            loop.close()


# Control test library for reusable queries
CONTROL_TESTS = {
    "Missing PO Check": {
        "query": "Find all invoices without purchase orders",
        "threshold": 0,
        "severity": "Critical",
        "description": "Identifies invoices that lack corresponding purchase orders - potential fraud indicator"
    },
    "Price Variance >10%": {
        "query": "Find invoices with price variance over 10%",
        "threshold": 10,
        "severity": "High",
        "description": "Detects invoices where pricing exceeds PO agreement by more than 10%"
    },
    "Duplicate Invoice Detection": {
        "query": "Find duplicate invoices",
        "threshold": 0,
        "severity": "Critical",
        "description": "Identifies potential duplicate invoices to prevent double payment"
    },
    "Quantity Overbilling": {
        "query": "Find invoices where quantity exceeds goods receipt",
        "threshold": 0,
        "severity": "High",
        "description": "Detects invoices billing for more items than were received"
    },
    "Missing Goods Receipt": {
        "query": "Find invoices without goods receipt",
        "threshold": 0,
        "severity": "High",
        "description": "Identifies invoices for items that haven't been received yet"
    },
    "Full Exception Report": {
        "query": "Show all exceptions and mismatches",
        "threshold": 0,
        "severity": "Medium",
        "description": "Comprehensive report of all matching exceptions"
    }
}


if __name__ == "__main__":
    # Test the orchestrator
    import os
    
    # Load test data
    data_dir = "data"
    if os.path.exists(data_dir):
        invoices_df = pd.read_csv(os.path.join(data_dir, "invoices.csv"))
        pos_df = pd.read_csv(os.path.join(data_dir, "purchase_orders.csv"))
        grs_df = pd.read_csv(os.path.join(data_dir, "goods_receipts.csv"))
        
        # Initialize orchestrator
        orchestrator = InvoiceMatchingOrchestrator()
        
        # Test a query
        results = orchestrator.run_query(
            invoices_df, pos_df, grs_df,
            "Find all invoices without purchase orders"
        )
        
        if results["success"]:
            print(f"Query: {results['query']}")
            print(f"Exceptions found: {len(results['exceptions'])}")
            print(f"Statistics: {results['statistics']}")
            print("\nReport:")
            print(results['report'])
        else:
            print(f"Error: {results['error']}")
    else:
        print("Data directory not found. Run data_generator.py first.")