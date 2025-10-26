import re
import json
import pandas as pd
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Set
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken, MessageContext, RoutedAgent, message_handler

class MatchingRequest(BaseModel):
    """Request to perform three-way matching"""
    invoices_df: Any  # Instead of pd.DataFrame
    pos_df: Any
    grs_df: Any
    mappings: Optional[Dict] = None
    query: Optional[str] = None

class MatchingResponse(BaseModel):
    """Response from matching operation"""
    matched: Any
    exceptions: Any
    statistics: Dict
    mappings_used: str
    vendor_risk: Optional[Any] = None
    prepared_inv: Optional[Any] = None

class AnalysisRequest(BaseModel):
    """Request for pattern analysis"""
    exceptions_df: Any
    matched_df: Any
    vendor_risk_df: Optional[Any] = None
    prepared_inv: Optional[Any] = None
    query: Optional[str] = None

class AnalysisResponse(BaseModel):
    """Response containing analysis results"""
    vendor_risk: Any
    temporal_patterns: Optional[Any] = None
    analysis: str
    summary: Dict
    patterns: List[str]
    recommendations: List[str]
    fraud_indicators: List[str]

class ReportRequest(BaseModel):
    """Request to generate audit report"""
    matching_results: dict
    analysis_results: dict
    mappings: dict
    company_name: Optional[str] = "Organization"
    auditor_name: Optional[str] = "Automated Audit System"
    query: Optional[str] = None

class ReportResponse(BaseModel):
    """Generated report response"""
    markdown_report: str
    summary: dict
    metadata: dict

class QueryRequest(BaseModel):
    """Request to execute a natural language query"""
    query: str
    invoices_df: Any
    pos_df: Any
    grs_df: Any
    mappings: Optional[Dict] = None

class QueryResponse(BaseModel):
    """Response from query execution"""
    results: Any  # DataFrame
    query_explanation: str
    generated_code: str
    row_count: int

class DataMapperAgent:
    """
    A specialized AutoGen agent designed to analyze multiple procurement data schemas 
    (Invoices, POs, GRs) and return a structured mapping dictionary.
    """

    def __init__(self, model: str = 'gemini-2.5-flash', api_key: str = None):
        """Initializes the Agent with the model client."""
        if not api_key:
            raise ValueError("API Key not found. Please set GEMINI_API_KEY in your .env file.")
            
        self.model_client = OpenAIChatCompletionClient(
            model=model,
            api_key=api_key
        )
        self.agent = None # Will be initialized in analyze_schemas
        self.data_schemas: Dict[str, Any] = {}

    # Private Helpers
    def _summarize_input_df(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extracts schema summary (columns, types, nulls, samples)."""
        columns = df.columns.tolist()
        types = {k: str(v) for k, v in df.dtypes.to_dict().items()}
        null_counts = {k: int(v) for k, v in df.isnull().sum().to_dict().items()}
        
        # NOTE: Using head(3) for sample values
        return {
            "columns": columns,
            "types": types,
            "sample_size": len(df),
            "null_counts": null_counts,
            "sample_values": {
                col: df[col].head(3).tolist() for col in df.columns[:]
            }
        }

    def _build_llm_prompt(self) -> str:
        """Constructs the structured prompt using the stored schemas."""
        # Schema dictionary -> clean indented JSON string
        schema_json_string = json.dumps(self.data_schemas, indent=2)
        schema_json_string = schema_json_string[:5000]  # ~ 4-5 KB Max
        
        # Note: Using f-string for template insertion, {{ and }} for literal braces
        prompt_template = f"""
        You are an expert **Data Engineer** specializing in finance and procurement system integration.
        Your task is to analyze three related data schemas (Invoices, Purchase Orders, Goods Receipts) and provide a **structured mapping** for downstream data validation and joining.

        **INSTRUCTIONS:**
        1.  Analyze the 'DATA SCHEMAS' JSON block provided below.
        2.  **Identify the following columns in each DataFrame:**
            -   **Join Keys:** The common columns that link the tables together (e.g., 'PO_Number' or 'Vendor_ID').
            -   **Validation Columns:** Columns representing quantities, amounts/prices, and dates that need to be compared across systems.
            -   **ID Columns:** Primary keys and foreign keys for individual transactions (Invoice IDs, PO IDs, GR IDs) and master data (Vendor IDs, Item/SKU Codes).
        3.  **Identify Data Quality Issues:** Look at null counts and sample values to flag potential issues (e.g., high null counts in a critical column, mixed data types, or inconsistent date formats).
        4.  **Confidence Score:** Assign a confidence score (0.0 to 1.0) based on the clarity of column names and the completeness of the data.

        **IMPORTANT COLUMN NAME PATTERNS:**
        -   "rate", "inv_rate", "po_rate", "unit_rate" → These are PRICE columns
        -   "units", "inv_units", "recv_units", "ordered_units" → These are QUANTITY columns  
        -   "net_amt", "gross_amt", "total_value", "amt" → These are AMOUNT columns
        -   Be flexible with abbreviations: supp_cd = supplier code, itm_cd = item code, dt = date, recv = received

        **CONSTRAINTS (CRITICAL):**
        -   You **MUST output ONLY a single JSON object**.
        -   Do **NOT** include any commentary, explanation, or conversational text.
        -   Enclose the final JSON object in a **Markdown code block**, like this: ```json{{...}}```

        **OUTPUT STRUCTURE:**
        The JSON object **MUST** strictly adhere to this format:
        {{
            "join_keys": {{
                "invoice_to_po": "...",  // The key (column name) linking Invoices to POs
                "po_to_gr": "...",       // The key (column name) linking POs to GRs
                "vendor_check": "..."    // The key (column name) used to link all three for a vendor check
            }},
            "validation_columns": {{
                "quantities": [{{"df": "invoices_df", "column": "Invoice_Qty"}}, ...],
                "amounts": [{{"df": "invoices_df", "column": "Unit_Price"}}, ...],
                "dates": [{{"df": "invoices_df", "column": "Invoice_Date"}}, ...]
            }},
            "identified_columns": {{
                "invoice_ids": ["..."],  // List of column names
                "po_ids": ["..."],       // List of column names
                "gr_ids": ["..."],       // List of column names
                "vendor_ids": ["..."],   // List of column names
                "item_codes": ["..."]    // List of column names
            }},
            "data_quality_issues": ["...", "..."], // List of text descriptions of issues
            "confidence": x.x // Float value
        }}

        ---
        **DATA SCHEMAS (JSON):**
        {schema_json_string}
        ---
        """
        return prompt_template.strip()

    def _json_parser(self, text_response: str) -> Dict[str, Any]:
        """Extracts JSON using multiple fallbacks."""
        cleaned_response = text_response.strip()

        # Strategy 1: Markdown Code Block
        try:
            match = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned_response)
            if match:
                json_string = match.group(1)
                print("Parser Status: Successfully extracted JSON from Markdown block.")
                return json.loads(json_string)
        except json.JSONDecodeError as e: 
            print(f"Warning: Markdown block failed to parse: {e}")
            pass

        # Strategy 2: Direct JSON Parse
        try:
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                print("Parser Status: Successfully parsed as clean, direct JSON.")
                return json.loads(cleaned_response)
        except json.JSONDecodeError as e:
            print(f"Warning: Direct JSON parse failed: {e}")
            pass

        # Strategy 3: Fuzzy Substring Extraction 
        try:
            start = cleaned_response.find('{')
            end = cleaned_response.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_substring = cleaned_response[start:end+1]
                json_substring = json_substring.replace(',]', ']').replace(',}', '}')
                print("Parser Status: Successfully extracted and parsed JSON substring (Fuzzy Mode).")
                return json.loads(json_substring)
        except json.JSONDecodeError as e:
            print(f"Warning: Fuzzy substring extraction failed: {e}")
            pass

        # Strategy 4: Fallback to Smart Defaults
        print("CRITICAL: All JSON parsing failed. Generating smart defaults.")
        return self._generate_smart_defaults()

    def _generate_smart_defaults(self) -> Dict[str, Any]:
        """Heuristically infers basic data mappings."""

        inv_cols = set(self.data_schemas.get('invoices_df', {}).get('columns', []))
        pos_cols = set(self.data_schemas.get('pos_df', {}).get('columns', []))
        grs_cols = set(self.data_schemas.get('grs_df', {}).get('columns', []))

        # Helper for Join Key Inference
        def find_best_join_key(candidates: Set[str]) -> str:
            candidates_upper = {c.upper() for c in candidates}
            if 'PO_ID' in candidates_upper:
                return next(c for c in candidates if c.upper() == 'PO_ID')
            if any('PO' in c.upper() for c in candidates):
                return next(c for c in candidates if 'PO' in c.upper())
            return list(candidates)[0] if candidates else 'PO_ID_INFERRED'

        po_join_candidates = inv_cols.intersection(pos_cols)
        gr_join_candidates = pos_cols.intersection(grs_cols)
        
        po_join = find_best_join_key(po_join_candidates)
        gr_join = find_best_join_key(gr_join_candidates)

        vendor_cols_candidates = inv_cols.intersection(pos_cols).union(grs_cols)
        vendor_key = next((col for col in vendor_cols_candidates if 'vendor' in col.lower() or 'supplier' in col.lower()), 'Vendor_ID_INFERRED')
        
        # Helper for Validation Column Formatting
        def find_and_format_cols(df_name: str, columns: Set[str], keywords: List[str]):
            return [{"df": df_name, "column": col} 
                    for col in columns 
                    if any(k in col.lower() for k in keywords)]

        inferred_quantities = []
        inferred_amounts = []
        inferred_dates = []
        
        df_map = {
            'invoices_df': inv_cols, 
            'pos_df': pos_cols, 
            'grs_df': grs_cols # Uses 'grs_df' key
        }
        
        for df_name, cols in df_map.items():
            inferred_quantities.extend(find_and_format_cols(df_name, cols, ['qty', 'quantity']))
            inferred_amounts.extend(find_and_format_cols(df_name, cols, ['price', 'amount', 'cost']))
            inferred_dates.extend(find_and_format_cols(df_name, cols, ['date', 'dt']))
            
        default_ids = ["Invoice_ID_FALLBACK", "PO_ID_FALLBACK", "GR_ID_FALLBACK", "Vendor_ID_FALLBACK", "Item_Code_FALLBACK"]
        
        # Construct Final Default Dictionary
        return {
            "join_keys": {
                "invoice_to_po": po_join,
                "po_to_gr": gr_join,
                "vendor_check": vendor_key
            },
            "validation_columns": {
                "quantities": inferred_quantities or [{"df": "inferred", "column": "None Found"}],
                "amounts": inferred_amounts or [{"df": "inferred", "column": "None Found"}],
                "dates": inferred_dates or [{"df": "inferred", "column": "None Found"}]
            },
            "identified_columns": {
                "invoice_ids": [next((c for c in inv_cols if 'invoice_id' in c.lower()), default_ids[0])],
                "po_ids": [next((c for c in pos_cols if 'po_id' in c.lower()), default_ids[1])],
                "gr_ids": [next((c for c in grs_cols if 'gr_id' in c.lower()), default_ids[2])],
                "vendor_ids": [vendor_key],
                "item_codes": [next((c for c in inv_cols.union(pos_cols) if 'item_code' in c.lower()), default_ids[4])]
            },
            "data_quality_issues": ["Mapping based on smart defaults (LLM response failed to parse). Review manually."],
            "confidence": 0.4 if "inferred" in vendor_key.lower() else 0.7
,
        }

    # Public Interface
    async def analyze_schemas(self, invoices_df: pd.DataFrame, pos_df: pd.DataFrame, grs_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Public method to run the entire schema analysis and mapping process.
        Returns the final structured mapping dictionary.
        """
        
        # Ingest and Summarize Input DataFrames
        invoice_summary = self._summarize_input_df(invoices_df)
        pos_summary = self._summarize_input_df(pos_df)
        grs_summary = self._summarize_input_df(grs_df)
        
        # Combine into a single structure for LLM Prompt
        self.data_schemas = {
            "invoices_df": invoice_summary,
            "pos_df": pos_summary,
            "grs_df": grs_summary
        }

        # Construct a Structured Prompt for the LLM
        system_prompt = self._build_llm_prompt()

        # Initialize Agent with System Prompt
        self.agent = AssistantAgent(
            name="DataMapperAgent",
            model_client=self.model_client,
            system_message=system_prompt
        )

        # Call the LLM
        task_prompt = "Analyze the provided data schemas and return the complete data mapping object strictly following the JSON format and constraints defined in your system prompt."
        
        response = await self.agent.run(task=task_prompt)
        raw_llm_output = response.messages[-1].content
        
        # Parse the LLM's Response
        parsed_response = self._json_parser(raw_llm_output)

        # Return Final Mapping Dictionary
        return parsed_response

class MatchingAgent(RoutedAgent):
    """Agent responsible for executing three-way matching logic
    
    This agent ONLY performs matching operations and basic filtering.
    It does NOT generate insights or analyze patterns.
    """
    
    def __init__(self, model_client, description: str = "Three-Way Matcher"):
        super().__init__(description)
        self.model_client = model_client
        self._matcher = None
    
    @property
    def matcher(self):
        """Lazy load ThreeWayMatcher"""
        if self._matcher is None:
            from matcher import ThreeWayMatcher
            self._matcher = ThreeWayMatcher()
        return self._matcher
    
    @message_handler
    async def handle_matching_request(
        self, 
        message: MatchingRequest, 
        ctx: MessageContext
    ) -> MatchingResponse:
        """Handle matching requests from other agents"""
        result = await self.execute_matching(
            invoices_df=message.invoices_df,
            pos_df=message.pos_df,
            grs_df=message.grs_df,
            mappings=message.mappings,
            query=message.query,
            cancellation_token=ctx.cancellation_token
        )
        
        return MatchingResponse(
            matched=result['matched'],
            exceptions=result['exceptions'],
            statistics=result['statistics'],
            mappings_used=result['mappings_used']
        )
    
    async def execute_matching(
        self,
        invoices_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        grs_df: pd.DataFrame,
        mappings: Optional[Dict] = None,
        query: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> Dict:
        """Execute matching based on query or full analysis
        
        NOTE: This method now ONLY performs matching and filtering.
        It does NOT generate insights - that's AnalysisAgent's job.
        """
        
        # Run the actual matching logic WITH mappings from DataMapper
        results = self.matcher.match(
            invoices_df, pos_df, grs_df,
            provided_mappings=mappings
        )
        
        # If there's a specific query, filter results using intelligent query parsing
        if query:
            filtered_exceptions = await self._filter_by_query_intelligent(
                results.exceptions_df, 
                query,
                cancellation_token=cancellation_token
            )
        else:
            filtered_exceptions = results.exceptions_df
        
        # Return raw matching results WITHOUT insights
        return {
            "matched": results.matched_df,
            "exceptions": filtered_exceptions,
            "statistics": results.statistics,
            "vendor_risk": results.vendor_risk_df,
            "prepared_inv": results.prepared_inv,
            "mappings_used": mappings.get('note', 'DataMapper mappings applied') if mappings else 'Auto-detected mappings'
        }
    
    async def _filter_by_query_intelligent(
        self, 
        exceptions_df: pd.DataFrame, 
        query: str,
        cancellation_token: Optional[CancellationToken] = None
    ) -> pd.DataFrame:
        """Use LLM to intelligently interpret query and generate filter logic"""
        
        if exceptions_df.empty:
            return exceptions_df
    
        # Get available filter options from the data
        available_exception_types = exceptions_df['type'].unique().tolist() if 'type' in exceptions_df.columns else []
        available_vendors = exceptions_df['vendor'].unique().tolist()[:20] if 'vendor' in exceptions_df.columns else []
        
        # Get amount range
        if 'financial_impact' in exceptions_df.columns:
            min_amount = float(exceptions_df['financial_impact'].min())
            max_amount = float(exceptions_df['financial_impact'].max())
        else:
            min_amount = max_amount = 0
        
        # Create filter interpretation prompt
        filter_prompt = f"""
        You are a query interpreter for an invoice auditing system. 
                
        User query: "{query}"

        Available data for filtering:
        - Exception Types: {available_exception_types}
        - Vendor IDs (sample): {available_vendors[:10]}
        - Financial Impact Range: ${min_amount:,.2f} to ${max_amount:,.2f}
        - Total exceptions: {len(exceptions_df)}

        Your task: Interpret the user's query and return a JSON object with filter criteria.

        Return ONLY a valid JSON object with this structure:
        {{
            "exception_types": ["NO_PO", "DUPLICATE"] or null for all,
            "vendor_ids": ["V0001", "V0002"] or null for all,
            "min_amount": 1000.0 or null,
            "max_amount": 50000.0 or null,
            "reasoning": "Brief explanation of how you interpreted the query"
        }}

        Now interpret this query: "{query}"
        Return ONLY the JSON, no additional text."""

        try:
            # Create temporary agent for filtering
            temp_agent = AssistantAgent(
                name="FilterInterpreter",
                model_client=self.model_client,
                system_message="You interpret filtering queries and return JSON."
            )
            
            response = await temp_agent.on_messages(
                [TextMessage(content=filter_prompt, source="user")],
                cancellation_token=cancellation_token
            )
            
            # Parse the filter criteria
            filter_criteria = self._parse_filter_response(response.chat_message.content)
            
            # Apply filters programmatically
            filtered_df = self._apply_filters(exceptions_df, filter_criteria)
            
            return filtered_df
            
        except Exception as e:
            print(f"LLM filtering failed: {e}, falling back to basic filtering")
            return self._filter_by_query_basic(exceptions_df, query)
    
    def _parse_filter_response(self, response: str) -> Dict:
        """Parse LLM response into filter criteria"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        try:
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
        except:
            pass
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        return {
            "exception_types": None,
            "vendor_ids": None,
            "min_amount": None,
            "max_amount": None,
            "reasoning": "Failed to parse filter criteria"
        }
    
    def _apply_filters(self, exceptions_df: pd.DataFrame, filter_criteria: Dict) -> pd.DataFrame:
        """Apply filter criteria to exceptions dataframe"""
        filtered = exceptions_df.copy()
        
        if filter_criteria.get('exception_types'):
            if 'type' in filtered.columns:
                filtered = filtered[filtered['type'].isin(filter_criteria['exception_types'])]
        
        if filter_criteria.get('vendor_ids'):
            if 'vendor' in filtered.columns:
                filtered = filtered[filtered['vendor'].isin(filter_criteria['vendor_ids'])]
        
        if filter_criteria.get('min_amount') is not None:
            if 'financial_impact' in filtered.columns:
                filtered = filtered[filtered['financial_impact'] >= float(filter_criteria['min_amount'])]
        
        if filter_criteria.get('max_amount') is not None:
            if 'financial_impact' in filtered.columns:
                filtered = filtered[filtered['financial_impact'] <= float(filter_criteria['max_amount'])]
        
        return filtered
    
    def _filter_by_query_basic(self, exceptions_df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Fallback basic filtering using string matching"""
        if exceptions_df.empty:
            return exceptions_df
            
        query_lower = query.lower()
        
        if "without purchase order" in query_lower or "no po" in query_lower:
            return exceptions_df[exceptions_df['type'] == 'NO_PO']
        elif "price variance" in query_lower:
            return exceptions_df[exceptions_df['type'] == 'PRICE_VARIANCE']
        elif "duplicate" in query_lower:
            return exceptions_df[exceptions_df['type'] == 'DUPLICATE']
        elif "quantity" in query_lower or "qty" in query_lower:
            return exceptions_df[exceptions_df['type'] == 'QTY_MISMATCH']
        elif "goods receipt" in query_lower or "no gr" in query_lower:
            return exceptions_df[exceptions_df['type'] == 'NO_GR']
        
        return exceptions_df

class AnalysisAgent(RoutedAgent):
    """Agent responsible for pattern analysis and risk assessment
    
    This agent analyzes matching results to identify patterns, assess vendor risk,
    and provide actionable recommendations for audit findings.
    
    Responsibilities:
    - Pattern detection (vendor, temporal, systemic)
    - Vendor risk profiling
    - Fraud indicator detection
    - Actionable recommendations generation
    """
    
    def __init__(self, model_client, description: str = "Financial Audit Analyst"):
        super().__init__(description)
        self.model_client = model_client
        
        self.system_message = """
        You are a financial audit analyst specializing in accounts payable.
        Your responsibilities:
        1. Identify patterns in exceptions and mismatches
        2. Assess vendor risk profiles
        3. Calculate financial exposure by category
        4. Detect potential fraud patterns
        5. Prioritize exceptions by risk level

        Focus on:
        - Vendor behavior patterns (systematic issues vs isolated incidents)
        - Temporal patterns (timing of exceptions, seasonality)
        - Amount distributions (outliers, clustering)
        - Systemic vs isolated issues

        Provide actionable insights with specific recommendations."""
    
    @message_handler
    async def handle_analysis_request(
        self, 
        message: AnalysisRequest, 
        ctx: MessageContext
    ) -> AnalysisResponse:
        """Handle analysis requests from other agents"""
        result = await self.analyze_patterns(
            exceptions_df=message.exceptions_df,
            matched_df=message.matched_df,
            vendor_risk_df=message.vendor_risk_df,
            prepared_inv=message.prepared_inv,
            query=message.query,
            cancellation_token=ctx.cancellation_token
        )
        
        return AnalysisResponse(
            vendor_risk=result['vendor_risk'],
            temporal_patterns=result.get('temporal_patterns'),
            analysis=result['analysis'],
            summary=result['summary'],
            patterns=result.get('patterns', []),
            recommendations=result.get('recommendations', []),
            fraud_indicators=result.get('fraud_indicators', [])
        )
    
    async def analyze_patterns(
        self, 
        exceptions_df: pd.DataFrame, 
        matched_df: pd.DataFrame,
        vendor_risk_df: Optional[pd.DataFrame] = None,
        prepared_inv: Optional[pd.DataFrame] = None,
        query: Optional[str] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> Dict:
        """Analyze patterns in matching results"""
        
        # Handle empty exceptions case
        if exceptions_df.empty:
            return {
                "vendor_risk": pd.DataFrame(),
                "temporal_patterns": None,
                "patterns": [],
                "recommendations": ["No exceptions found. All invoices matched successfully."],
                "analysis": "✅ Clean matching - no issues detected.",
                "summary": {
                    "total_exceptions": 0, 
                    "total_exposure": 0,
                    "exception_types": {},
                    "high_risk_vendors": {},
                    "critical_exceptions": 0,
                    "high_exceptions": 0
                },
                "fraud_indicators": []
            }
        
        # Vendor Risk Analysis
        vendor_risk = self._analyze_vendor_risk(
            exceptions_df, 
            vendor_risk_df,
            prepared_inv
        )
        
        # Temporal Analysis
        temporal_patterns = self._analyze_temporal_patterns(exceptions_df, prepared_inv)
        
        # Fraud Detection
        fraud_indicators = self._detect_fraud_patterns(exceptions_df, prepared_inv)
        
        # Create Analysis Summary
        analysis_summary = self._create_analysis_summary(
            exceptions_df, 
            matched_df,
            vendor_risk,
            fraud_indicators
        )
        
        # Get LLM Analysis
        llm_analysis = await self._get_llm_analysis(
            analysis_summary, 
            query,
            cancellation_token
        )
        
        # Extract patterns and recommendations
        patterns, recommendations = self._parse_llm_response(llm_analysis)
        
        return {
            "vendor_risk": vendor_risk,
            "temporal_patterns": temporal_patterns,
            "analysis": llm_analysis,
            "summary": analysis_summary,
            "patterns": patterns,
            "recommendations": recommendations,
            "fraud_indicators": fraud_indicators
        }
    
    def _analyze_vendor_risk(
        self, 
        exceptions_df: pd.DataFrame,
        vendor_risk_df: Optional[pd.DataFrame],
        prepared_inv: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        """Vendor risk analysis with behavioral patterns"""
        
        if vendor_risk_df is not None and not vendor_risk_df.empty:
            # Use existing vendor risk from matcher
            vendor_risk = vendor_risk_df.copy()
        else:
            # Calculate from scratch
            if 'vendor' not in exceptions_df.columns:
                return pd.DataFrame()
            
            vendor_risk = exceptions_df.groupby('vendor').agg(
                total_exceptions=('exception_id', 'count'),
                total_impact=('financial_impact', 'sum')
            ).reset_index()
            
            vendor_risk['risk_score'] = (
                vendor_risk['total_exceptions'] * 10 + 
                vendor_risk['total_impact'] / 1000
            )
            vendor_risk['risk_level'] = pd.cut(
                vendor_risk['risk_score'], 
                bins=[0, 50, 150, 300, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        
        # Add exception type distribution per vendor
        if 'type' in exceptions_df.columns:
            exception_dist = exceptions_df.groupby(['vendor', 'type']).size().unstack(fill_value=0)
            vendor_risk = vendor_risk.merge(exception_dist, on='vendor', how='left')
        
        return vendor_risk.sort_values('risk_score', ascending=False)
    
    def _analyze_temporal_patterns(
        self, 
        exceptions_df: pd.DataFrame,
        prepared_inv: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Analyze temporal patterns in exceptions"""
        
        # Try to find date column from multiple sources
        date_col = None
        source_df = exceptions_df
        
        # First, try exceptions_df
        for col in ['date', 'invoice_date', 'Date', 'Invoice_Date', 'timestamp']:
            if col in exceptions_df.columns:
                date_col = col
                break
        
        # If not found, try prepared_inv
        if date_col is None and prepared_inv is not None:
            for col in prepared_inv.columns:
                if 'date' in col.lower() or 'dt' in col.lower():
                    # Map back to exceptions using invoice_ref
                    if '_row_ref' in prepared_inv.columns and 'invoice_ref' in exceptions_df.columns:
                        date_map = prepared_inv.set_index('_row_ref')[col].to_dict()
                        exceptions_df['_temp_date'] = exceptions_df['invoice_ref'].map(date_map)
                        date_col = '_temp_date'
                        source_df = exceptions_df
                        break
        
        if date_col is None:
            return None
        
        try:
            temp_df = source_df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col], errors='coerce')
            valid_dates = temp_df.dropna(subset=[date_col])
            
            if valid_dates.empty or 'type' not in valid_dates.columns:
                return None
            
            # Group by week and exception type
            temporal_patterns = valid_dates.groupby([
                pd.Grouper(key=date_col, freq='W'),
                'type'
            ]).size().unstack(fill_value=0)
            
            return temporal_patterns
            
        except Exception as e:
            print(f"Temporal analysis failed: {e}")
            return None
    
    def _detect_fraud_patterns(
        self, 
        exceptions_df: pd.DataFrame,
        prepared_inv: Optional[pd.DataFrame]
    ) -> List[str]:
        """Detect potential fraud indicators"""
        fraud_indicators = []
        
        # High rate of NO_PO exceptions (potential ghost invoices)
        if 'type' in exceptions_df.columns:
            no_po_count = (exceptions_df['type'] == 'NO_PO').sum()
            no_po_rate = no_po_count / len(exceptions_df) if len(exceptions_df) > 0 else 0
            
            if no_po_rate > 0.3:  # More than 30% NO_PO
                fraud_indicators.append(
                    f"⚠️ HIGH NO_PO RATE: {no_po_rate:.1%} of exceptions lack purchase orders (potential ghost invoices)"
                )
        
        # Duplicate invoices (double payment attempt)
        if 'type' in exceptions_df.columns:
            dup_count = (exceptions_df['type'] == 'DUPLICATE').sum()
            if dup_count > 0:
                fraud_indicators.append(
                    f"🔴 DUPLICATES DETECTED: {dup_count} duplicate invoices found (double payment risk)"
                )
        
        # Systematic price variances from specific vendors
        if 'type' in exceptions_df.columns and 'vendor' in exceptions_df.columns:
            price_var_vendors = exceptions_df[
                exceptions_df['type'] == 'PRICE_VARIANCE'
            ]['vendor'].value_counts()
            
            for vendor, count in price_var_vendors.head(3).items():
                if count >= 3:  # 3+ price variances from same vendor
                    fraud_indicators.append(
                        f"⚠️ SYSTEMATIC OVERBILLING: Vendor '{vendor}' has {count} price variance exceptions"
                    )
        
        # Large financial impacts
        if 'financial_impact' in exceptions_df.columns:
            high_impact = exceptions_df[exceptions_df['financial_impact'] > 10000]
            if not high_impact.empty:
                total_high_impact = high_impact['financial_impact'].sum()
                fraud_indicators.append(
                    f"💰 HIGH VALUE AT RISK: ${total_high_impact:,.2f} in exceptions over $10,000"
                )
        
        return fraud_indicators
    
    def _create_analysis_summary(
        self, 
        exceptions_df: pd.DataFrame,
        matched_df: pd.DataFrame,
        vendor_risk: pd.DataFrame,
        fraud_indicators: List[str]
    ) -> Dict:
        """Create structured summary for LLM analysis"""
        
        summary = {
            "total_invoices": len(matched_df) + len(exceptions_df),
            "total_matched": len(matched_df),
            "total_exceptions": len(exceptions_df),
            "match_rate": f"{(len(matched_df) / (len(matched_df) + len(exceptions_df)) * 100):.1f}%",
            "total_exposure": 0,
            "exception_types": {},
            "high_risk_vendors": {},
            "critical_exceptions": 0,
            "high_exceptions": 0,
            "fraud_indicators_count": len(fraud_indicators)
        }
        
        # Financial exposure
        if 'financial_impact' in exceptions_df.columns:
            summary["total_exposure"] = float(exceptions_df['financial_impact'].sum())
        
        # Exception type breakdown
        if 'type' in exceptions_df.columns:
            summary["exception_types"] = exceptions_df['type'].value_counts().to_dict()
        
        # High risk vendors
        if not vendor_risk.empty and 'risk_level' in vendor_risk.columns:
            high_risk = vendor_risk[vendor_risk['risk_level'].isin(['High', 'Critical'])]
            if not high_risk.empty:
                summary["high_risk_vendors"] = high_risk.head(5).to_dict('records')
            
            summary["critical_exceptions"] = int((vendor_risk['risk_level'] == 'Critical').sum())
            summary["high_exceptions"] = int((vendor_risk['risk_level'] == 'High').sum())
        
        return summary
    
    async def _get_llm_analysis(
        self, 
        analysis_summary: Dict, 
        query: Optional[str],
        cancellation_token: Optional[CancellationToken]
    ) -> str:
        """Get LLM analysis of the summary data"""
        
        query_context = f"\n\nSpecific Focus: {query}" if query else ""
        
        prompt = f"""
        Analyze these audit findings and provide actionable insights:

        ## Summary Statistics
        {json.dumps(analysis_summary, default=str, indent=2)}
        {query_context}

        ## Required Analysis:

        1. **Top 3-5 Risk Patterns**: Identify the most significant patterns in the exceptions
        2. **Vendor-Specific Concerns**: Highlight vendors requiring immediate attention
        3. **Process Improvements**: Recommend specific process changes to prevent future exceptions
        4. **Fraud Indicators**: Flag any potential fraud or systematic abuse patterns
        5. **Priority Actions**: List 3-5 immediate actions prioritized by impact

        Format your response with clear sections and bullet points."""
        
        try:
            temp_agent = AssistantAgent(
                name="AnalysisAssistant",
                model_client=self.model_client,
                system_message=self.system_message
            )
            
            response = await temp_agent.on_messages(
                [TextMessage(content=prompt, source="user")],
                cancellation_token=cancellation_token
            )
            
            return response.chat_message.content
            
        except Exception as e:
            print(f"LLM analysis failed: {e}")
            return self._generate_fallback_analysis(analysis_summary)
    
    def _generate_fallback_analysis(self, summary: Dict) -> str:
        """Generate basic analysis if LLM fails"""
        parts = [
            "## Analysis Summary\n",
            f"📊 **Total Invoices**: {summary['total_invoices']}",
            f"✅ **Matched**: {summary['total_matched']} ({summary['match_rate']})",
            f"⚠️ **Exceptions**: {summary['total_exceptions']}",
            f"💰 **Financial Exposure**: ${summary['total_exposure']:,.2f}\n"
        ]
        
        if summary['exception_types']:
            parts.append("\n## Exception Breakdown:")
            for exc_type, count in summary['exception_types'].items():
                parts.append(f"- **{exc_type}**: {count}")
        
        if summary['critical_exceptions'] > 0:
            parts.append(
                f"\n🔴 **CRITICAL**: {summary['critical_exceptions']} critical-risk vendors require immediate action"
            )
        
        return "\n".join(parts)
    
    def _parse_llm_response(self, llm_response: str) -> tuple[List[str], List[str]]:
        """Extract patterns and recommendations from LLM response"""
        patterns = []
        recommendations = []
        
        lines = llm_response.split('\n')
        in_patterns = False
        in_recommendations = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Section detection
            if any(kw in line.lower() for kw in ['pattern', 'risk', 'concern']):
                in_patterns = True
                in_recommendations = False
                continue
            elif any(kw in line.lower() for kw in ['recommendation', 'action', 'improvement']):
                in_patterns = False
                in_recommendations = True
                continue
            
            # Extract items
            if line.startswith(('-', '*', '•', '●')) or (len(line) > 2 and line[0].isdigit() and line[1] in '.):'):
                item = line.lstrip('-*•●0123456789.): ')
                if in_patterns and item:
                    patterns.append(item)
                elif in_recommendations and item:
                    recommendations.append(item)
        
        # Fallback
        if not patterns:
            patterns = ["See full analysis for pattern details"]
        if not recommendations:
            recommendations = ["Review high-risk vendors", "Focus on critical exceptions"]
        
        return patterns, recommendations

class ReportAgent(RoutedAgent):
    """
    LLM-powered audit report generator.
    Uses structured context (stats, exceptions, risks, analysis)
    and delegates narrative + formatting to the model.
    """

    def __init__(self, model_client: OpenAIChatCompletionClient, description: str = "Audit Report Generator"):
        super().__init__(description)
        self.model = model_client

    @message_handler
    async def handle_report_request(self, message: ReportRequest, ctx: MessageContext) -> ReportResponse:
        """Generate audit report from matching and analysis results"""

        # Extract data
        stats = message.matching_results.get("statistics", {})
        exceptions_df = message.matching_results.get("exceptions", pd.DataFrame())
        matched_df = message.matching_results.get("matched", pd.DataFrame())
        vendor_risk = message.matching_results.get("vendor_risk")

        analysis_summary = message.analysis_results.get("summary", {})
        patterns = message.analysis_results.get("patterns", [])
        recommendations = message.analysis_results.get("recommendations", [])
        fraud_indicators = message.analysis_results.get("fraud_indicators", [])
        analysis_text = message.analysis_results.get("analysis", "")

        # Use LLM to generate the markdown report
        llm_report = await self._generate_llm_report(
            company_name=message.company_name,
            auditor_name=message.auditor_name,
            stats=stats,
            exceptions_df=exceptions_df,
            vendor_risk=vendor_risk,
            analysis_summary=analysis_summary,
            patterns=patterns,
            recommendations=recommendations,
            fraud_indicators=fraud_indicators,
            analysis_text=analysis_text,
            custom_focus=message.query,
            cancellation_token=ctx.cancellation_token
        )

        # local summary + metadata
        summary = {
            "total_invoices": stats.get("total_invoices", 0),
            "match_rate": stats.get("match_rate_pct", 0),
            "total_exceptions": stats.get("total_exceptions", 0),
            "financial_exposure": stats.get("financial_exposure", 0),
            "high_risk_vendors": analysis_summary.get("high_exceptions", 0),
            "critical_issues": len(fraud_indicators),
        }

        metadata = {
            "generated_at": datetime.now().isoformat(),
            "report_type": "Three-Way Matching Audit",
            "data_sources": ["Invoices", "Purchase Orders", "Goods Receipts"],
        }

        return ReportResponse(
            markdown_report=llm_report,
            summary=summary,
            metadata=metadata,
        )
    
    async def _generate_llm_report(
        self,
        company_name: str,
        auditor_name: str,
        stats: dict,
        exceptions_df: pd.DataFrame,
        vendor_risk: Optional[pd.DataFrame],
        analysis_summary: dict,
        patterns: list,
        recommendations: list,
        fraud_indicators: list,
        analysis_text: str,
        custom_focus: Optional[str],
        cancellation_token: Any
    ) -> str:
        """Use LLM to synthesize and format full markdown report"""

        # Prepare structured context
        context = {
            "company_name": company_name,
            "auditor_name": auditor_name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "custom_focus": custom_focus,
            "statistics": stats,
            "analysis_summary": analysis_summary,
            "patterns": patterns,
            "recommendations": recommendations,
            "fraud_indicators": fraud_indicators,
            "vendor_risk": vendor_risk.head(10).to_dict("records") if vendor_risk is not None and not vendor_risk.empty else [],
            "top_exceptions": exceptions_df.nlargest(10, 'financial_impact').to_dict("records") if not exceptions_df.empty and 'financial_impact' in exceptions_df.columns else [],
            "exception_breakdown": stats.get('exception_breakdown', {}),
            "analysis_text": analysis_text,
        }

        # Prompt for the LLM
        system_message = """You are a professional financial audit report writer.

        Your task is to generate comprehensive, executive-level audit reports in Markdown format.

        ## Report Structure (MUST include all sections):
        1. **Executive Summary** - Overall status, quick facts, key findings
        2. **Key Performance Metrics** - Statistics tables with match rates, exception rates
        3. **Vendor Risk Assessment** - High-risk vendors table (if any)
        4. **Exception Analysis** - Breakdown by type and top exceptions by financial impact
        5. **Identified Patterns** - Risk patterns from analysis
        6. **Fraud Indicators** - Critical alerts (if any)
        7. **Recommendations** - Actionable next steps
        8. **Detailed Analysis** - In-depth narrative analysis
        9. **Appendix** - Methodology, thresholds, metadata

        ## Formatting Guidelines:
        - Use tables for numeric data (| Header | Value |)
        - Use bullet points for lists
        - Use emojis for visual clarity (✅ ⚠️ 🚨)
        - Use bold for emphasis
        - Keep tone professional but clear
        - Highlight critical issues prominently

        ## Status Indicators:
        - Match rate ≥95% + no fraud = ✅ GOOD
        - Match rate ≥85% + ≤2 fraud indicators = ⚠️ ATTENTION REQUIRED  
        - Otherwise = 🚨 CRITICAL

        Generate a complete, professional audit report."""

        user_prompt = f"""Generate a comprehensive financial audit report using the following data:

        ```json
        {json.dumps(context, indent=2, default=str)}
        ```

        Create a professional Markdown report following all the guidelines provided."""

        # Call LLM via OpenAIChatCompletionClient
        try:
            # Create proper message objects instead of dicts
            messages = [
                SystemMessage(content=system_message),
                UserMessage(content=user_prompt, source="user")
            ]
            
            # Call the create method with proper message format
            response = await self.model.create(
                messages=messages,
                cancellation_token=cancellation_token
            )

            markdown_report = response.content
            
            return markdown_report
            
        except Exception as e:
            # Fallback report if LLM fails
            return f"""# Financial Audit Report - Generation Error

            **Error:** Failed to generate report via LLM: {str(e)}

            ## Summary Data
            - Total Invoices: {stats.get('total_invoices', 0)}
            - Match Rate: {stats.get('match_rate_pct', 0):.1f}%
            - Exceptions: {stats.get('total_exceptions', 0)}
            - Financial Exposure: ${stats.get('financial_exposure', 0):,.2f}

            Please check logs for details."""

class QueryAgent(RoutedAgent):
    """Agent that translates natural language queries to pandas operations
    
    Uses mappings from DataMapperAgent to understand column structure.
    Helps users explore raw data before running three-way matching.
    """
    
    def __init__(self, model_client, description: str = "Data Query Interpreter"):
        super().__init__(description)
        self.model_client = model_client
        
        self.system_message = """
        You are a data query interpreter that translates natural language questions into pandas code.

        ## Your Task:
        Convert user questions into working pandas code that filters or analyzes the data.

        ## Available DataFrames:
        - invoices_df: Invoice records
        - pos_df: Purchase Order records  
        - grs_df: Goods Receipt records

        ## Response Format:
        1. Brief explanation of what you're doing (1 sentence)
        2. Working pandas code in a ```python``` code block
        3. Store the final result in a variable called `result`

        ## Example:

        User: "Show me invoices over $5000"

        Explanation: Filtering invoices where amount exceeds $5000

        Code:
        ```python
        result = invoices_df[invoices_df['Invoice_Amount'] > 5000]
        ```

        ## Rules:
        - Use only pandas operations (no imports, no file operations)
        - Always store final result in `result` variable
        - Use column names exactly as provided in the context
        - Keep code simple and readable
        """
    
    @message_handler
    async def handle_query_request(
        self, 
        message: QueryRequest, 
        ctx: MessageContext
    ) -> QueryResponse:
        """Handle query requests"""
        result = await self.execute_query(
            query=message.query,
            invoices_df=message.invoices_df,
            pos_df=message.pos_df,
            grs_df=message.grs_df,
            mappings=message.mappings,
            cancellation_token=ctx.cancellation_token
        )
        
        return QueryResponse(
            results=result['results'],
            query_explanation=result['explanation'],
            generated_code=result['code'],
            row_count=result['row_count']
        )
    
    async def execute_query(
        self,
        query: str,
        invoices_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        grs_df: pd.DataFrame,
        mappings: Optional[Dict] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> Dict[str, Any]:
        """Execute a natural language query on the data"""
        
        # Build context with columns and mappings
        context = self._build_query_context(
            invoices_df, pos_df, grs_df, mappings
        )
        
        # Create prompt
        prompt = f"""
        {context}

        User Query: "{query}"

        Generate pandas code to answer this query. Use the column names and mappings provided above.
        """
        
        try:
            # Get LLM to generate code
            temp_agent = AssistantAgent(
                name="QueryInterpreter",
                model_client=self.model_client,
                system_message=self.system_message
            )
            
            response = await temp_agent.on_messages(
                [TextMessage(content=prompt, source="user")],
                cancellation_token=cancellation_token
            )
            
            # Parse response
            parsed = self._parse_llm_response(response.chat_message.content)
            
            # Execute code
            result_df = self._execute_pandas_code(
                parsed['code'],
                invoices_df,
                pos_df,
                grs_df
            )
            
            return {
                'results': result_df,
                'explanation': parsed['explanation'],
                'code': parsed['code'],
                'row_count': len(result_df)
            }
                
        except Exception as e:
            return {
                'results': pd.DataFrame(),
                'explanation': f"Query failed: {str(e)}",
                'code': "",
                'row_count': 0
            }
    
    def _build_query_context(
        self,
        invoices_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        grs_df: pd.DataFrame,
        mappings: Optional[Dict]
    ) -> str:
        """Build context with available columns and mappings"""
        
        context_parts = [
            "## Available Data:\n",
            f"invoices_df ({len(invoices_df)} rows):",
            f"Columns: {', '.join(invoices_df.columns.tolist())}\n",
            f"pos_df ({len(pos_df)} rows):",
            f"Columns: {', '.join(pos_df.columns.tolist())}\n",
            f"grs_df ({len(grs_df)} rows):",
            f"Columns: {', '.join(grs_df.columns.tolist())}\n"
        ]
        
        # Add mapping information from DataMapperAgent
        if mappings:
            context_parts.append("\n## Column Mappings (from DataMapperAgent):")
            
            # Key columns
            if mappings.get('invoice_id'):
                context_parts.append(f"- Invoice ID column: {mappings['invoice_id']}")
            if mappings.get('po_id'):
                context_parts.append(f"- PO ID column: {mappings['po_id']}")
            if mappings.get('vendor'):
                context_parts.append(f"- Vendor column: {mappings['vendor']}")
            if mappings.get('item'):
                context_parts.append(f"- Item column: {mappings['item']}")
            
            # Amount columns
            if mappings.get('inv_amount'):
                context_parts.append(f"- Invoice amount column: {mappings['inv_amount']}")
            if mappings.get('inv_price'):
                context_parts.append(f"- Invoice price column: {mappings['inv_price']}")
            if mappings.get('po_price'):
                context_parts.append(f"- PO price column: {mappings['po_price']}")
            
            # Quantity columns
            if mappings.get('inv_quantity'):
                context_parts.append(f"- Invoice quantity column: {mappings['inv_quantity']}")
            if mappings.get('po_quantity'):
                context_parts.append(f"- PO quantity column: {mappings['po_quantity']}")
            if mappings.get('gr_quantity'):
                context_parts.append(f"- GR quantity column: {mappings['gr_quantity']}")
            
            # Date columns
            if mappings.get('date'):
                context_parts.append(f"- Date column: {mappings['date']}")
            
            context_parts.append("\nUse these mapped columns when filtering or analyzing data.\n")
        
        return "\n".join(context_parts)
    
    def _parse_llm_response(self, response: str) -> Dict[str, str]:
        """Extract explanation and code from LLM response"""
        
        # Extract code block
        code_match = re.search(r'```python\s*\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            # Try without language specifier
            code_match = re.search(r'```\s*\n(.*?)\n```', response, re.DOTALL)
            code = code_match.group(1).strip() if code_match else ""
        
        # Extract explanation (text before code block)
        if code_match:
            explanation = response[:code_match.start()].strip()
        else:
            explanation = response.strip()
        
        # Clean up explanation
        explanation = explanation.split('\n')[0] if explanation else "Filtering data based on query"
        
        return {
            'explanation': explanation,
            'code': code
        }
    
    def _execute_pandas_code(
        self,
        code: str,
        invoices_df: pd.DataFrame,
        pos_df: pd.DataFrame,
        grs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Execute pandas code and return result"""
        
        # Create namespace with dataframes
        namespace = {
            'invoices_df': invoices_df.copy(),
            'pos_df': pos_df.copy(),
            'grs_df': grs_df.copy(),
            'pd': pd,
            'result': None
        }
        
        try:
            # Execute code
            exec(code, namespace)
            
            result = namespace.get('result')
            
            if result is None:
                return pd.DataFrame()
            
            # Convert to DataFrame if needed
            if not isinstance(result, pd.DataFrame):
                try:
                    result = pd.DataFrame(result)
                except:
                    return pd.DataFrame()
            
            return result
            
        except Exception as e:
            print(f"Execution error: {e}")
            return pd.DataFrame()
