import re
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# Config + Result dataclasses
@dataclass
class MatchConfig:
    price_variance_threshold: float = 0.10
    qty_tolerance: float = 0.02
    duplicate_amount_round: int = 2
    use_fuzzy_vendor: bool = False
    fuzzy_threshold: int = 85

@dataclass
class MatchResults:
    matched_df: pd.DataFrame
    exceptions_df: pd.DataFrame
    statistics: Dict[str, Any]
    vendor_risk_df: Optional[pd.DataFrame] = None
    prepared_inv: Optional[pd.DataFrame] = None

# Helper Utilities
def _norm_text(x: Any) -> str:
    if pd.isna(x):
        return ""
    return re.sub(r'\s+',' ', str(x)).strip().lower()

def _norm_id(x: Any) -> str:
    """Normalize identifiers: uppercase, remove non-alphanum except -/_"""
    if pd.isna(x):
        return ""
    s = str(x).upper()
    s = re.sub(r'[^A-Z0-9\-_]', '', s)
    return s

def _ensure_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors='coerce').fillna(0.0)

# Matching Engine
class ThreeWayMatcher:
    def __init__(self, config: Optional[MatchConfig]=None):
        self.cfg = config or MatchConfig()

    def resolve_mappings(
            self,
            provided: Optional[Dict],
            invoices: pd.DataFrame,
            pos: pd.DataFrame,
            grs: pd.DataFrame
    ) -> Dict:
        """
        Takes the parsed JSON (LLM Output) and returns a simplified dict.
        Handles the nested dict format from DataMapperAgent.
        """
        def detect_cols(df: pd.DataFrame):
            cols = {c.lower(): c for c in df.columns}
            mapping = {}
            for k in cols:
                if 'invoice' in k and any(x in k for x in ['id', 'no', 'number', 'num']):
                    mapping['invoice_id'] = cols[k]
                if 'po' in k and any(x in k for x in ['id','no','number','num']):
                    mapping['po_id'] = cols[k]
                if 'gr' in k and any(x in k for x in ['id','no','number','num']):
                    mapping['gr_id'] = cols[k]
                if any(x in k for x in ['vendor','supplier','seller']):
                    mapping['vendor'] = cols[k]
                if any(x in k for x in ['item','sku','product','material']):
                    mapping['item'] = cols[k]
                if any(x in k for x in ['qty','quantity','units']):
                    mapping.setdefault('quantity', cols[k])
                if any(x in k for x in ['price','unit_price','rate','cost','unit_cost','inv_rate','po_rate','unit_rate']):
                    mapping.setdefault('price', cols[k])
                if any(x in k for x in ['amount','total','invoice_amount','value']):
                    mapping.setdefault('amount', cols[k])
                if 'date' in k or 'dt' in k:
                    mapping.setdefault('date', cols[k])
            
            return mapping

        resolved = {}
        if provided:
            ids = provided.get('identified_columns', {})
            jk = provided.get('join_keys', {})
            vc = provided.get('validation_columns', {})
            
            # Invoice ID
            if ids.get('invoice_ids'):
                resolved['invoice_id'] = ids['invoice_ids'][0]
            if ids.get('po_ids'):
                resolved['po_id'] = ids['po_ids'][0]
            if ids.get('gr_ids'):
                resolved['gr_id'] = ids['gr_ids'][0]
            if ids.get('vendor_ids'):
                resolved['vendor'] = ids['vendor_ids'][0]
            if ids.get('item_codes'):
                resolved['item'] = ids['item_codes'][0]

            # Join Keys
            if jk.get('invoice_to_po'):
                resolved['po_join'] = jk['invoice_to_po']
            if jk.get('po_to_gr'):
                resolved['gr_join'] = jk['po_to_gr']
            # For vendor check, we'll use the vendor join key if available
            if jk.get('vendor_check'):
                resolved['vendor_join'] = jk['vendor_check']

            # Validation Columns
            if vc.get('quantities'):
                for qty_entry in vc['quantities']:
                    if isinstance(qty_entry, dict):
                        df_name = qty_entry.get('df', '')
                        col_name = qty_entry.get('column')
                        
                        if 'invoices' in df_name and col_name:
                            resolved['inv_quantity'] = col_name
                        elif 'pos' in df_name and col_name:
                            resolved['po_quantity'] = col_name
                        elif 'grs' in df_name and col_name:
                            resolved['gr_quantity'] = col_name
                    else:
                        # Fallback: string format
                        if 'quantity' not in resolved:
                            resolved['quantity'] = qty_entry
            
            if vc.get('amounts'):
                for amt_entry in vc['amounts']:
                    if isinstance(amt_entry, dict):
                        df_name = amt_entry.get('df', '')
                        col_name = amt_entry.get('column')
                        
                        if 'invoices' in df_name and col_name:
                            if 'price' in col_name.lower() and 'inv_price' not in resolved:
                                resolved['inv_price'] = col_name
                            elif 'amount' in col_name.lower() and 'inv_amount' not in resolved:
                                resolved['inv_amount'] = col_name
                        elif 'pos' in df_name and col_name:
                            if 'price' in col_name.lower() and 'po_price' not in resolved:
                                resolved['po_price'] = col_name
                            elif 'amount' in col_name.lower() and 'po_amount' not in resolved:
                                resolved['po_amount'] = col_name
                    else:
                        # Fallback: string format
                        if 'price' not in resolved:
                            resolved['price'] = amt_entry

        # Fallback detection
        inv_map = detect_cols(invoices)
        po_map = detect_cols(pos)
        gr_map = detect_cols(grs)
        
        # Merge with fallbacks
        for key, m in [('invoice_id', inv_map), ('po_id', po_map), ('gr_id', gr_map),
                    ('vendor', inv_map), ('item', inv_map)]:
            if key not in resolved:
                resolved[key] = m.get(key)
        
        # DataFrame-specific fallbacks
        resolved.setdefault('inv_quantity', inv_map.get('quantity'))
        resolved.setdefault('po_quantity', po_map.get('quantity'))
        resolved.setdefault('gr_quantity', gr_map.get('quantity'))
        resolved.setdefault('inv_price', inv_map.get('price'))
        resolved.setdefault('po_price', po_map.get('price'))
        resolved.setdefault('inv_amount', inv_map.get('amount'))
        
        # Fallbacks for backward compatibility
        resolved.setdefault('quantity', resolved.get('inv_quantity'))
        resolved.setdefault('price', resolved.get('inv_price'))
        resolved.setdefault('amount', resolved.get('inv_amount'))
        
        # Join key fallbacks
        resolved.setdefault('po_join', resolved.get('po_id'))
        resolved.setdefault('gr_join', resolved.get('po_join'))  # Use PO join for GR since they typically share PO_ID
        
        # Final defensive defaults
        for k in ['invoice_id','po_id','gr_id','vendor','item',
                  'inv_quantity','po_quantity','gr_quantity',
                  'inv_price','po_price','inv_amount',
                  'quantity','price','amount','date','po_join','gr_join','vendor_join']:
            resolved.setdefault(k, None)
            
        return resolved
    
    def _prepare_tables(
            self,
            invoices: pd.DataFrame,
            pos: pd.DataFrame,
            grs: pd.DataFrame,
            mapping: Dict
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Standardizes and normalizes input DataFrames."""
        inv = invoices.copy()
        po = pos.copy()
        gr = grs.copy()

        # Invoice ID
        if mapping['invoice_id'] and mapping['invoice_id'] in inv.columns:
            inv['_invoice_id'] = inv[mapping['invoice_id']].apply(_norm_id)
        else:
            inv['_invoice_id'] = inv.index.astype(str)

        # PO join keys - use the join key from mapping
        po_key = mapping['po_join']
        if po_key and po_key in inv.columns:
            inv['_po_join'] = inv[po_key].apply(_norm_id)
        elif mapping['po_id'] and mapping['po_id'] in inv.columns:
            inv['_po_join'] = inv[mapping['po_id']].apply(_norm_id)
        else:
            inv['_po_join'] = inv.index.astype(str)

        if mapping['po_join'] and mapping['po_join'] in po.columns:
            po['_po_join'] = po[mapping['po_join']].apply(_norm_id)
        elif mapping['po_id'] and mapping['po_id'] in po.columns:
            po['_po_join'] = po[mapping['po_id']].apply(_norm_id)
        else:
            po['_po_join'] = po.index.astype(str)

        # GR join - use the same key as PO since GR typically has the same ID as PO
        if mapping['gr_join'] and mapping['gr_join'] in gr.columns:
            gr['_gr_join'] = gr[mapping['gr_join']].apply(_norm_id)
        elif mapping['po_join'] and mapping['po_join'] in gr.columns:
            gr['_gr_join'] = gr[mapping['po_join']].apply(_norm_id)
        elif mapping['po_id'] and mapping['po_id'] in gr.columns:
            gr['_gr_join'] = gr[mapping['po_id']].apply(_norm_id)
        else:
            gr['_gr_join'] = gr.index.astype(str)

        # Vendor normalization
        for df_obj, df_name in [(inv, 'inv'), (po, 'po'), (gr, 'gr')]:
            vendor_col = mapping.get('vendor')
            if vendor_col and vendor_col in df_obj.columns:
                df_obj['_vendor_norm'] = df_obj[vendor_col].apply(_norm_text)
            else:
                df_obj['_vendor_norm'] = ""

        # Item normalization
        for df_obj, df_name in [(inv, 'inv'), (po, 'po'), (gr, 'gr')]:
            item_col = mapping.get('item')
            if item_col and item_col in df_obj.columns:
                df_obj['_item_join'] = df_obj[item_col].apply(_norm_id)
            else:
                df_obj['_item_join'] = ""

        # Numeric conversions - USE DATAFRAME-SPECIFIC COLUMNS
        if mapping.get('inv_quantity') and mapping['inv_quantity'] in inv.columns:
            inv['_qty_val'] = _ensure_numeric(inv[mapping['inv_quantity']])
        else:
            inv['_qty_val'] = 0.0

        if mapping.get('po_quantity') and mapping['po_quantity'] in po.columns:
            po['_qty_val'] = _ensure_numeric(po[mapping['po_quantity']])
        else:
            po['_qty_val'] = 0.0

        if mapping.get('gr_quantity') and mapping['gr_quantity'] in gr.columns:
            gr['_qty_val'] = _ensure_numeric(gr[mapping['gr_quantity']])
        else:
            gr['_qty_val'] = 0.0

        # Price - invoice and PO only
        if mapping.get('inv_price') and mapping['inv_price'] in inv.columns:
            inv['_price_val'] = _ensure_numeric(inv[mapping['inv_price']])
        else:
            inv['_price_val'] = 0.0

        if mapping.get('po_price') and mapping['po_price'] in po.columns:
            po['_price_val'] = _ensure_numeric(po[mapping['po_price']])
        else:
            po['_price_val'] = 0.0

        # Amount - invoice only
        if mapping.get('inv_amount') and mapping['inv_amount'] in inv.columns:
            inv['_amount_val'] = _ensure_numeric(inv[mapping['inv_amount']])
        else:
            # Calculate amount if not provided
            inv['_amount_val'] = inv['_qty_val'] * inv['_price_val']

        # Provenance
        inv['_row_ref'] = inv.index.map(lambda x: f"inv::{x}")
        po['_row_ref'] = po.index.map(lambda x: f"po::{x}")
        gr['_row_ref'] = gr.index.map(lambda x: f"gr::{x}")

        return inv, po, gr
        
    def detect_duplicates(self, inv: pd.DataFrame, mapping: Dict) -> Tuple[pd.DataFrame, List[Dict]]:
        """Identify duplicates using fuzzy amount matching to catch near-identical invoices
        
        Duplicate criteria:
        - Same PO_ID
        - Same Vendor
        - Amount within 2% (to catch intentional slight variations)
        - Optionally same item code for stricter matching
        """
        exceptions = []
        
        if '_po_join' not in inv.columns or '_vendor_norm' not in inv.columns or '_amount_val' not in inv.columns:
            return inv, []
        
        # Track which invoices are duplicates
        duplicate_refs = set()
        
        # Group by PO + Vendor to find potential duplicates
        grouped = inv.groupby(['_po_join', '_vendor_norm'])
        
        for (po_id, vendor), group in grouped:
            if len(group) < 2:
                continue  # Need at least 2 invoices to have duplicates
            
            # Get amounts and references
            group_data = []
            for idx, row in group.iterrows():
                group_data.append({
                    'index': idx,
                    'row_ref': row['_row_ref'],
                    'amount': row['_amount_val'],
                    'item': row.get('_item_join', ''),
                    'qty': row.get('_qty_val', 0)
                })
            
            # Compare each pair of invoices in the group
            for i in range(len(group_data)):
                for j in range(i + 1, len(group_data)):
                    inv_i = group_data[i]
                    inv_j = group_data[j]
                    
                    # Calculate amount difference percentage
                    avg_amount = (inv_i['amount'] + inv_j['amount']) / 2
                    if avg_amount == 0:
                        continue
                    
                    amount_diff_pct = abs(inv_i['amount'] - inv_j['amount']) / avg_amount
                    
                    # Check if amounts are within 2% of each other
                    if amount_diff_pct <= 0.02:
                        # Additional check: same item code (if available)
                        same_item = (inv_i['item'] == inv_j['item']) if inv_i['item'] and inv_j['item'] else True
                        
                        if same_item:
                            # Mark both as duplicates
                            for inv_data in [inv_i, inv_j]:
                                if inv_data['row_ref'] not in duplicate_refs:
                                    duplicate_refs.add(inv_data['row_ref'])
                                    
                                    exceptions.append({
                                        'exception_id': f"exc_dup::{inv_data['row_ref']}",
                                        'invoice_ref': inv_data['row_ref'],
                                        'type': 'DUPLICATE',
                                        'reason': f'Duplicate detected: PO={po_id}, Vendor={vendor}, Amount≈${inv_data["amount"]:.2f} (±{amount_diff_pct:.1%})',
                                        'financial_impact': float(inv_data['amount']),
                                        'evidence': {
                                            'po_id': str(po_id),
                                            'vendor': str(vendor),
                                            'amount': float(inv_data['amount']),
                                            'amount_variance_pct': float(amount_diff_pct),
                                            'duplicate_pair': True
                                        }
                                    })
        
        # Mark duplicates in the dataframe
        inv['_is_duplicate'] = inv['_row_ref'].isin(duplicate_refs)
        
        print(f"   Duplicate Detection: Found {len(exceptions)} duplicate invoices across {len(duplicate_refs)} unique refs")
        
        return inv, exceptions
    
    def match(
            self,
            invoices: pd.DataFrame,
            pos: pd.DataFrame,
            grs: pd.DataFrame,
            provided_mappings: Optional[Dict]=None
    ) -> MatchResults:
        """Main API: runs three-way matching."""

        mapping = self.resolve_mappings(provided_mappings, invoices, pos, grs)
        inv, po, gr = self._prepare_tables(invoices, pos, grs, mapping)

        all_exceptions: List[Dict] = []
        matched_records: List[Dict] = []

        # Duplicates
        inv, dup_excs = self.detect_duplicates(inv, mapping)
        all_exceptions.extend(dup_excs)

        # Join Invoices -> POs
        # Use both PO join and item join for better matching
        # Explicitly specify suffixes to avoid confusion
        po_subset = po[['_po_join', '_item_join', '_vendor_norm', '_row_ref', '_price_val', '_qty_val']].copy()
        po_subset = po_subset.rename(columns={'_row_ref': '_row_ref_po'})

        inv_po = inv.merge(
            po_subset,
            left_on=['_po_join', '_item_join'],
            right_on=['_po_join', '_item_join'],
            how='left',
            suffixes=('_inv','_po')
        )

        # After merge, the original invoice columns will have _inv suffix
        # The PO columns will have _po suffix
        # Ensure we have the correct references
        if '_row_ref_inv' not in inv_po.columns:
            # Fallback: the original _row_ref column should now be _row_ref_inv
            # But if not, we map from the original inv DataFrame
            inv_po['_row_ref_inv'] = inv_po['_row_ref']

        # No PO
        no_po_mask = inv_po['_row_ref_po'].isna()
        no_po_rows = inv_po[no_po_mask]
        for _, r in no_po_rows.iterrows():
            all_exceptions.append({
                'exception_id': f"exc_nopo::{r['_row_ref_inv']}",
                'invoice_ref': r['_row_ref_inv'],
                'type': 'NO_PO',
                'reason': 'No matching PO found for invoice',
                'financial_impact': float(r['_amount_val_inv'] if '_amount_val_inv' in r else r.get('_amount_val', 0)),
                'evidence': {'po_join': r['_po_join'], 'item_join': r['_item_join']}
            })
        
        inv_with_po = inv_po[~no_po_mask].copy()

        # Join with GRs; Rename _qty_val to avoid collision
        gr_subset = gr[['_gr_join', '_item_join', '_row_ref', '_qty_val']].copy()
        gr_subset = gr_subset.rename(
            columns={
                '_row_ref': '_row_ref_gr',
                '_qty_val': '_qty_val_gr'  # imp change
            }
        )

        inv_po_gr = inv_with_po.merge(
            gr_subset,
            left_on=['_po_join', '_item_join'],
            right_on=['_gr_join', '_item_join'],
            how='left'
        )

        # After the second merge, ensure all required columns exist
        # The invoice columns have _inv suffix, PO columns have _po suffix, GR columns have _gr suffix
        # Original invoice columns that didn't collide keep their suffixes
        if '_row_ref_inv' not in inv_po_gr.columns:
            inv_po_gr['_row_ref_inv'] = inv_po_gr['_row_ref']

        # No GR
        no_gr_mask = inv_po_gr['_row_ref_gr'].isna()
        no_gr_rows = inv_po_gr[no_gr_mask]
        for _, r in no_gr_rows.iterrows():
            all_exceptions.append({
                'exception_id': f"exc_nogr::{r['_row_ref_inv']}",
                'invoice_ref': r['_row_ref_inv'],
                'type': 'NO_GR',
                'reason': 'No GR found for matched PO',
                'financial_impact': float(r['_amount_val_inv'] if '_amount_val_inv' in r else r.get('_amount_val', 0)),
                'evidence': {'po_join': r['_po_join'], 'item_join': r['_item_join']}
            })

        inv_with_gr = inv_po_gr[~no_gr_mask].copy()

        # Vectorized checks: Use the correct column names with suffixes
        # inv_with_gr now has:
        # - Invoice columns: _qty_val_inv, _price_val_inv, _amount_val_inv, _row_ref_inv
        # - PO columns: _qty_val_po, _price_val_po, _row_ref_po  
        # - GR columns: _qty_val_gr, _row_ref_gr
        
        # Handle division by zero in quantity variance calculation
        inv_with_gr['_qty_variance_rel'] = np.where(
            inv_with_gr['_qty_val_gr'] != 0,
            (inv_with_gr['_qty_val_inv'] - inv_with_gr['_qty_val_gr']) / inv_with_gr['_qty_val_gr'],
            np.inf  # Use infinity to indicate division by zero
        )
        
        # Handle division by zero in price variance calculation
        inv_with_gr['_price_variance_rel'] = np.where(
            inv_with_gr['_price_val_po'] != 0,
            (inv_with_gr['_price_val_inv'] - inv_with_gr['_price_val_po']).abs() / inv_with_gr['_price_val_po'],
            np.inf  # Use infinity to indicate division by zero
        )

        # Quantity checks
        qty_mask = inv_with_gr['_qty_val_inv'] > (inv_with_gr['_qty_val_gr'] * (1 + self.cfg.qty_tolerance))
        partial_mask = (inv_with_gr['_qty_val_inv'] > 0) & (inv_with_gr['_qty_val_inv'] < inv_with_gr['_qty_val_gr'])
        price_mask = inv_with_gr['_price_variance_rel'] > self.cfg.price_variance_threshold

        for _, r in inv_with_gr.iterrows():
            # Check for quantity mismatches (overbilling)
            if qty_mask.loc[r.name]:
                all_exceptions.append({
                    'exception_id': f"exc_qty::{r['_row_ref_inv']}",
                    'invoice_ref': r['_row_ref_inv'],
                    'type': 'QTY_MISMATCH',
                    'reason': f"Invoice qty {r['_qty_val_inv']} > received {r['_qty_val_gr']}",
                    'financial_impact': float(max(0, r['_qty_val_inv'] - r['_qty_val_gr']) * r['_price_val_inv']),
                    'evidence': {'invoice_qty': r['_qty_val_inv'], 'gr_qty': r['_qty_val_gr']}
                })
            elif partial_mask.loc[r.name]:
                # Don't add exception for partial receipt - it's just informational
                pass
            
            # Check for price variance
            if price_mask.loc[r.name]:
                all_exceptions.append({
                    'exception_id': f"exc_price::{r['_row_ref_inv']}",
                    'invoice_ref': r['_row_ref_inv'],
                    'type': 'PRICE_VARIANCE',
                    'reason': f"Price deviation {r['_price_variance_rel']:.2%}",
                    'financial_impact': float(abs(r['_qty_val_inv'] * (r['_price_val_inv'] - r['_price_val_po']))),
                    'evidence': {'invoice_price': r['_price_val_inv'], 'po_price': r['_price_val_po']}
                })
            
            # Mark as matched only if no critical exceptions (duplicates, no PO, no GR, qty mismatch, price variance)
            is_duplicate = r.get('_is_duplicate', False)
            if not is_duplicate and not qty_mask.loc[r.name] and not price_mask.loc[r.name]:
                matched_records.append({
                    'invoice_ref': r['_row_ref_inv'],
                    'type': 'MATCHED',
                    'reason': 'All checks passed',
                    'financial_impact': float(r['_amount_val_inv'] if '_amount_val_inv' in r else r.get('_amount_val', 0))
                })

        # Build DataFrames
        exceptions_df = pd.DataFrame(all_exceptions)
        matched_df = pd.DataFrame(matched_records)

        # Stats
        total_invoices = len(invoices)
        total_matched = len(matched_df)
        total_exceptions = len(exceptions_df)
        stats = {
            'total_invoices': total_invoices,
            'total_matched': total_matched,
            'total_exceptions': total_exceptions,
            'match_rate_pct': (total_matched/total_invoices*100) if total_invoices else 0,
            'exception_rate_pct': (total_exceptions/total_invoices*100) if total_invoices else 0,
            'financial_exposure': exceptions_df['financial_impact'].sum() if not exceptions_df.empty else 0.0,
            'exception_breakdown': exceptions_df['type'].value_counts().to_dict() if not exceptions_df.empty else {}
        }

        # Vendor risk
        vendor_risk_df = None
        if not exceptions_df.empty:
            vendor_map = inv.set_index('_row_ref')['_vendor_norm'].to_dict()
            exceptions_df['vendor'] = exceptions_df['invoice_ref'].map(vendor_map)
            
            vendor_group = exceptions_df.groupby('vendor').agg(
                total_exceptions=('exception_id', 'count'),
                total_impact=('financial_impact', 'sum')
            ).reset_index()
            vendor_group['risk_score'] = (
                vendor_group['total_exceptions'] * 10 + 
                vendor_group['total_impact'] / 1000
            )
            vendor_group['risk_level'] = pd.cut(
                vendor_group['risk_score'], 
                bins=[0, 50, 150, 300, float('inf')],  # Added Critical tier
                labels=['Low', 'Medium', 'High', 'Critical']
            )
            vendor_risk_df = vendor_group.sort_values('risk_score', ascending=False)

        return MatchResults(
            matched_df=matched_df, 
            exceptions_df=exceptions_df, 
            statistics=stats, 
            vendor_risk_df=vendor_risk_df,
            prepared_inv=inv
        )