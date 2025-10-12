"""
Three-Way Matching Logic for Invoice Processing
Implements the core matching logic between Invoices, Purchase Orders, and Goods Receipts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

class ThreeWayMatcher:
    """Core matching engine for three-way invoice matching"""
    
    def __init__(self, price_variance_threshold: float = 0.10, qty_variance_threshold: float = 0.05):
        """
        Initialize the matcher with configurable thresholds
        
        Args:
            price_variance_threshold: Maximum acceptable price variance (default 10%)
            qty_variance_threshold: Maximum acceptable quantity variance (default 5%)
        """
        self.price_variance_threshold = price_variance_threshold
        self.qty_variance_threshold = qty_variance_threshold
        self.reason_codes = {
            'NO_PO': 'Invoice without purchase order',
            'NO_GR': 'Invoice without goods receipt',
            'QTY_MISMATCH': 'Invoice quantity exceeds received quantity',
            'PRICE_VARIANCE': 'Price variance exceeds threshold',
            'DUPLICATE': 'Duplicate invoice detected',
            'PARTIAL_RECEIPT': 'Partial goods receipt',
            'MATCHED': 'Successfully matched'
        }
    
    def detect_column_mappings(self, df: pd.DataFrame, entity_type: str) -> Dict[str, str]:
        """
        Auto-detect column mappings based on column names and data patterns
        
        Args:
            df: DataFrame to analyze
            entity_type: Type of entity ('invoice', 'po', 'gr')
        
        Returns:
            Dictionary of standard column names to actual column names
        """
        columns = df.columns.tolist()
        mappings = {}
        
        # Common patterns for column detection
        patterns = {
            'id': ['_id', 'number', 'no', 'num', 'code'],
            'vendor': ['vendor', 'supplier', 'seller'],
            'item': ['item', 'product', 'material', 'sku'],
            'quantity': ['qty', 'quantity', 'amount', 'units'],
            'price': ['price', 'cost', 'rate', 'unit_price'],
            'amount': ['amount', 'total', 'value', 'sum'],
            'date': ['date', 'datetime', 'timestamp']
        }
        
        for col in columns:
            col_lower = col.lower()
            
            # Check for ID columns
            if entity_type == 'invoice' and 'invoice' in col_lower and any(p in col_lower for p in patterns['id']):
                mappings['invoice_id'] = col
            elif entity_type == 'po' and 'po' in col_lower and any(p in col_lower for p in patterns['id']):
                mappings['po_id'] = col
            elif entity_type == 'gr' and 'gr' in col_lower and any(p in col_lower for p in patterns['id']):
                mappings['gr_id'] = col
            
            # Check for common fields
            if 'po_id' in col_lower:
                mappings['po_id_ref'] = col
            elif any(p in col_lower for p in patterns['vendor']):
                mappings['vendor_id'] = col
            elif any(p in col_lower for p in patterns['item']):
                mappings['item_code'] = col
            elif any(p in col_lower for p in patterns['quantity']):
                mappings['quantity'] = col
            elif any(p in col_lower for p in patterns['price']):
                mappings['price'] = col
            elif any(p in col_lower for p in patterns['amount']):
                mappings['amount'] = col
            elif any(p in col_lower for p in patterns['date']):
                mappings['date'] = col
        
        return mappings
    
    def three_way_match(self, 
                       invoices_df: pd.DataFrame, 
                       pos_df: pd.DataFrame, 
                       grs_df: pd.DataFrame,
                       auto_map: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Perform three-way matching between invoices, purchase orders, and goods receipts
        
        Args:
            invoices_df: DataFrame containing invoice data
            pos_df: DataFrame containing purchase order data
            grs_df: DataFrame containing goods receipt data
            auto_map: Whether to auto-detect column mappings
        
        Returns:
            Tuple of (matched_df, exceptions_df, match_statistics)
        """
        # Create copies to avoid modifying original data
        invoices = invoices_df.copy()
        pos = pos_df.copy()
        grs = grs_df.copy()
        
        # Auto-detect column mappings if needed
        if auto_map:
            invoice_mappings = self.detect_column_mappings(invoices, 'invoice')
            po_mappings = self.detect_column_mappings(pos, 'po')
            gr_mappings = self.detect_column_mappings(grs, 'gr')
        
        # Initialize exception tracking
        exceptions = []
        matched_records = []
        
        # Step 1: Check for duplicate invoices
        duplicate_mask = invoices.duplicated(subset=['PO_ID', 'Invoice_Amount', 'Vendor_ID'], keep=False)
        duplicate_invoices = invoices[duplicate_mask].copy()
        for idx, invoice in duplicate_invoices.iterrows():
            exceptions.append({
                'Invoice_ID': invoice['Invoice_ID'],
                'Vendor_ID': invoice['Vendor_ID'],
                'PO_ID': invoice['PO_ID'],
                'Invoice_Amount': invoice['Invoice_Amount'],
                'Exception_Type': 'DUPLICATE',
                'Exception_Reason': self.reason_codes['DUPLICATE'],
                'Risk_Level': 'Critical',
                'Financial_Impact': invoice['Invoice_Amount']
            })
        
        # Remove duplicates for further processing (keep first occurrence)
        invoices_unique = invoices[~invoices.duplicated(subset=['PO_ID', 'Invoice_Amount', 'Vendor_ID'], keep='first')]
        
        # Step 2: Match invoices to POs
        invoices_with_po = invoices_unique.merge(
            pos[['PO_ID', 'Vendor_ID', 'Item_Code', 'Quantity', 'Unit_Price', 'Total_Amount']],
            on='PO_ID',
            how='left',
            suffixes=('_inv', '_po')
        )
        
        # Find invoices without PO
        no_po_mask = invoices_with_po['Vendor_ID_po'].isna()
        invoices_without_po = invoices_with_po[no_po_mask]
        
        for idx, invoice in invoices_without_po.iterrows():
            exceptions.append({
                'Invoice_ID': invoice['Invoice_ID'],
                'Vendor_ID': invoice['Vendor_ID_inv'],
                'PO_ID': invoice['PO_ID'],
                'Invoice_Amount': invoice['Invoice_Amount'],
                'Exception_Type': 'NO_PO',
                'Exception_Reason': self.reason_codes['NO_PO'],
                'Risk_Level': 'Critical',
                'Financial_Impact': invoice['Invoice_Amount']
            })
        
        # Continue with invoices that have PO
        invoices_with_valid_po = invoices_with_po[~no_po_mask]
        
        # Step 3: Match to Goods Receipts
        invoices_with_gr = invoices_with_valid_po.merge(
            grs[['PO_ID', 'GR_ID', 'Received_Qty', 'GR_Date']],
            on='PO_ID',
            how='left'
        )
        
        # Find invoices without GR
        no_gr_mask = invoices_with_gr['GR_ID'].isna()
        invoices_without_gr = invoices_with_gr[no_gr_mask]
        
        for idx, invoice in invoices_without_gr.iterrows():
            exceptions.append({
                'Invoice_ID': invoice['Invoice_ID'],
                'Vendor_ID': invoice['Vendor_ID_inv'],
                'PO_ID': invoice['PO_ID'],
                'Invoice_Amount': invoice['Invoice_Amount'],
                'Exception_Type': 'NO_GR',
                'Exception_Reason': self.reason_codes['NO_GR'],
                'Risk_Level': 'High',
                'Financial_Impact': invoice['Invoice_Amount']
            })
        
        # Continue with invoices that have both PO and GR
        complete_matches = invoices_with_gr[~no_gr_mask]
        
        # Step 4: Check quantity mismatches
        for idx, row in complete_matches.iterrows():
            qty_variance = (row['Invoice_Qty'] - row['Received_Qty']) / row['Received_Qty'] if row['Received_Qty'] > 0 else float('inf')
            
            if row['Invoice_Qty'] > row['Received_Qty']:
                exceptions.append({
                    'Invoice_ID': row['Invoice_ID'],
                    'Vendor_ID': row['Vendor_ID_inv'],
                    'PO_ID': row['PO_ID'],
                    'Invoice_Amount': row['Invoice_Amount'],
                    'Exception_Type': 'QTY_MISMATCH',
                    'Exception_Reason': f"{self.reason_codes['QTY_MISMATCH']} (Invoice: {row['Invoice_Qty']}, Received: {row['Received_Qty']})",
                    'Risk_Level': 'High',
                    'Financial_Impact': (row['Invoice_Qty'] - row['Received_Qty']) * row['Invoice_Price']
                })
            
            # Step 5: Check price variances
            price_variance = abs(row['Invoice_Price'] - row['Unit_Price']) / row['Unit_Price'] if row['Unit_Price'] > 0 else float('inf')
            
            if price_variance > self.price_variance_threshold:
                exceptions.append({
                    'Invoice_ID': row['Invoice_ID'],
                    'Vendor_ID': row['Vendor_ID_inv'],
                    'PO_ID': row['PO_ID'],
                    'Invoice_Amount': row['Invoice_Amount'],
                    'Exception_Type': 'PRICE_VARIANCE',
                    'Exception_Reason': f"{self.reason_codes['PRICE_VARIANCE']} ({price_variance:.1%} variance)",
                    'Risk_Level': 'Medium',
                    'Financial_Impact': row['Invoice_Qty'] * abs(row['Invoice_Price'] - row['Unit_Price'])
                })
            
            # If no exceptions, mark as matched
            if (row['Invoice_Qty'] <= row['Received_Qty'] and 
                price_variance <= self.price_variance_threshold):
                matched_records.append({
                    'Invoice_ID': row['Invoice_ID'],
                    'Vendor_ID': row['Vendor_ID_inv'],
                    'PO_ID': row['PO_ID'],
                    'Invoice_Amount': row['Invoice_Amount'],
                    'Match_Status': 'MATCHED',
                    'Match_Reason': self.reason_codes['MATCHED']
                })
        
        # Create DataFrames
        exceptions_df = pd.DataFrame(exceptions)
        matched_df = pd.DataFrame(matched_records)
        
        # Calculate statistics
        total_invoices = len(invoices_df)
        total_exceptions = len(exceptions_df)
        total_matched = len(matched_df)
        
        statistics = {
            'total_invoices': total_invoices,
            'total_matched': total_matched,
            'total_exceptions': total_exceptions,
            'match_rate': (total_matched / total_invoices * 100) if total_invoices > 0 else 0,
            'exception_rate': (total_exceptions / total_invoices * 100) if total_invoices > 0 else 0,
            'total_financial_exposure': exceptions_df['Financial_Impact'].sum() if len(exceptions_df) > 0 else 0,
            'exception_breakdown': exceptions_df['Exception_Type'].value_counts().to_dict() if len(exceptions_df) > 0 else {}
        }
        
        return matched_df, exceptions_df, statistics
    
    def analyze_vendor_risk(self, exceptions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze vendor risk based on exception patterns
        
        Args:
            exceptions_df: DataFrame containing exception records
        
        Returns:
            DataFrame with vendor risk analysis
        """
        if exceptions_df.empty:
            return pd.DataFrame()
        
        vendor_analysis = exceptions_df.groupby('Vendor_ID').agg({
            'Invoice_ID': 'count',
            'Financial_Impact': 'sum',
            'Exception_Type': lambda x: x.value_counts().to_dict()
        }).rename(columns={
            'Invoice_ID': 'Total_Exceptions',
            'Financial_Impact': 'Total_Financial_Impact'
        })
        
        # Calculate risk score
        vendor_analysis['Risk_Score'] = (
            vendor_analysis['Total_Exceptions'] * 10 +
            vendor_analysis['Total_Financial_Impact'] / 1000
        )
        
        # Assign risk level
        vendor_analysis['Risk_Level'] = pd.cut(
            vendor_analysis['Risk_Score'],
            bins=[0, 50, 150, float('inf')],
            labels=['Low', 'Medium', 'High']
        )
        
        return vendor_analysis.sort_values('Risk_Score', ascending=False)