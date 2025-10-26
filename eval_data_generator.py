import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os
import json

fake = Faker()
random.seed(42)
np.random.seed(42)

class EvaluationDataGenerator:
    def __init__(self, output_dir='eval_data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # Create eval_data folder
        
        # Creating vendors from v0100 to v0150
        self.vendors = [
            {'id': f'V{str(i).zfill(4)}', 'name': fake.company(), 'is_shell': False}
            for i in range(100, 151)  # Different vendor IDs from training
        ]
        
        # Make 5 shell companies
        for i in range(5):
            self.vendors[i]['is_shell'] = True
        
        self.items = [f'ITEM-{str(i).zfill(3)}' for i in range(200, 300)]
        self.approvers = [fake.name() for _ in range(8)]
        self.receivers = [fake.name() for _ in range(12)]
        
        # Approval threshold for fraud pattern
        self.approval_threshold = 10000
    
    def generate_date_range(self, start_days_ago=180, end_days_ago=0):
        """Generate random date within specified range"""
        start_date = datetime.now() - timedelta(days=start_days_ago)
        end_date = datetime.now() - timedelta(days=end_days_ago)
        random_days = random.randint(0, (end_date - start_date).days)
        return start_date + timedelta(days=random_days)
    
    def is_round_dollar(self, amount):
        """Check if amount is suspiciously round"""
        return amount % 1000 == 0 and amount >= 5000
    
    def is_just_below_threshold(self, amount):
        """Check if amount is just below approval threshold"""
        return 9500 <= amount < self.approval_threshold
    
    def generate_dataset_1_corporate(self):
        """Dataset 1: Corporate Standard Format"""
        print("\n=== Generating Dataset 1: Corporate Standard ===")
        
        # Different column names
        po_cols = ['PO_Num', 'VendorCode', 'VendorName', 'ItemCode', 'OrderQty', 
                   'UnitCost', 'OrderDate', 'Approver', 'Status', 'TotalAmt']
        gr_cols = ['GR_Num', 'PO_Num', 'ItemCode', 'ReceivedQty', 'ReceiptDate', 
                   'Receiver', 'VendorCode', 'Status']
        inv_cols = ['InvNo', 'VendorCode', 'VendorName', 'PO_Num', 'ItemCode', 
                    'InvQty', 'InvPrice', 'InvDate', 'Amt', 'Status']
        
        pos = []
        grs = []
        invoices = []
        ground_truth = []
        
        invoice_counter = 1
        
        # Generate 100 invoices with controlled patterns
        for i in range(100):
            vendor = random.choice(self.vendors)
            po_date = self.generate_date_range(180, 30)
            po_id = f'PO{str(1000 + i).zfill(6)}'
            
            # Determine fraud pattern for this invoice
            fraud_type = None
            if i < 60:  # 60 clean matches
                pattern = 'CLEAN'
            elif i < 70:  # 10 NO_PO (shell company pattern)
                pattern = 'NO_PO'
                vendor = random.choice([v for v in self.vendors if v['is_shell']])
                fraud_type = 'SHELL_COMPANY'
            elif i < 78:  # 8 PRICE_VARIANCE with round dollar scheme
                pattern = 'PRICE_VARIANCE'
                fraud_type = 'ROUND_DOLLAR'
            elif i < 85:  # 7 QTY_OVERBILLING
                pattern = 'QTY_OVERBILLING'
                fraud_type = 'SYSTEMATIC_OVERBILLING'
            elif i < 90:  # 5 DUPLICATE
                pattern = 'DUPLICATE'
            elif i < 95:  # 5 just-below-threshold
                pattern = 'PRICE_VARIANCE'
                fraud_type = 'BELOW_THRESHOLD'
            else:  # 5 NO_GR
                pattern = 'NO_GR'
            
            # Generate PO
            qty = random.randint(10, 500)
            
            if fraud_type == 'BELOW_THRESHOLD':
                # Price calculated to be just below threshold
                unit_price = random.uniform(9500, 9900) / qty
            elif fraud_type == 'ROUND_DOLLAR':
                # Round dollar unit price
                unit_price = random.choice([1000, 2000, 5000, 10000]) / qty
            else:
                unit_price = round(random.uniform(100, 5000), 2)
            
            po = {
                'PO_Num': po_id,
                'VendorCode': vendor['id'],
                'VendorName': vendor['name'],
                'ItemCode': random.choice(self.items),
                'OrderQty': qty,
                'UnitCost': unit_price,
                'OrderDate': po_date.strftime('%Y-%m-%d'),
                'Approver': random.choice(self.approvers),
                'Status': 'Approved',
                'TotalAmt': round(qty * unit_price, 2)
            }
            
            # Generate GR (if not NO_GR or NO_PO pattern)
            if pattern not in ['NO_GR', 'NO_PO']:
                gr_date = po_date + timedelta(days=random.randint(1, 25))
                received_qty = qty if pattern != 'QTY_OVERBILLING' else qty
                
                gr = {
                    'GR_Num': f'GR{str(1000 + len(grs)).zfill(6)}',
                    'PO_Num': po_id,
                    'ItemCode': po['ItemCode'],
                    'ReceivedQty': received_qty,
                    'ReceiptDate': gr_date.strftime('%Y-%m-%d'),
                    'Receiver': random.choice(self.receivers),
                    'VendorCode': vendor['id'],
                    'Status': 'Completed'
                }
                grs.append(gr)
                invoice_date = gr_date + timedelta(days=random.randint(1, 15))
            else:
                gr_date = po_date + timedelta(days=random.randint(5, 20))
                invoice_date = gr_date
                received_qty = qty
            
            # Generate Invoice based on pattern
            if pattern == 'NO_PO':
                # Ghost invoice - non-existent PO
                invoice_po = f'PO{str(9000 + i).zfill(6)}'
            else:
                invoice_po = po_id
                pos.append(po)
            
            if pattern == 'QTY_OVERBILLING':
                invoice_qty = received_qty + random.randint(10, 50)
            else:
                invoice_qty = received_qty
            
            if pattern == 'PRICE_VARIANCE':
                if fraud_type == 'ROUND_DOLLAR':
                    invoice_price = round(unit_price * 1.15, 2)  # 15% markup with rounding
                else:
                    invoice_price = round(unit_price * random.uniform(1.12, 1.25), 2)
            else:
                invoice_price = unit_price
            
            invoice = {
                'InvNo': f'INV{str(invoice_counter).zfill(6)}',
                'VendorCode': vendor['id'],
                'VendorName': vendor['name'],
                'PO_Num': invoice_po,
                'ItemCode': po['ItemCode'],
                'InvQty': invoice_qty,
                'InvPrice': invoice_price,
                'InvDate': invoice_date.strftime('%Y-%m-%d'),
                'Amt': round(invoice_qty * invoice_price, 2),
                'Status': 'Pending'
            }
            invoices.append(invoice)
            
            # Duplicate handling
            if pattern == 'DUPLICATE':
                duplicate = invoice.copy()
                invoice_counter += 1
                duplicate['InvNo'] = f'INV{str(invoice_counter).zfill(6)}'
                duplicate['InvQty'] = invoice_qty + random.randint(-2, 2)
                duplicate['InvPrice'] = round(invoice_price * random.uniform(0.99, 1.01), 2)
                duplicate['Amt'] = round(duplicate['InvQty'] * duplicate['InvPrice'], 2)
                duplicate['InvDate'] = (datetime.strptime(invoice['InvDate'], '%Y-%m-%d') + 
                                       timedelta(days=random.randint(-2, 3))).strftime('%Y-%m-%d')
                invoices.append(duplicate)
                
                # Ground truth for duplicate
                ground_truth.append({
                    'invoice_id': duplicate['InvNo'],
                    'expected_exception': 'DUPLICATE',
                    'is_fraud': False,
                    'fraud_type': None,
                    'financial_impact': duplicate['Amt'],
                    'should_match': False
                })
            
            # Ground truth for main invoice
            expected_exception = 'NONE' if pattern == 'CLEAN' else pattern
            ground_truth.append({
                'invoice_id': invoice['InvNo'],
                'expected_exception': expected_exception,
                'is_fraud': fraud_type is not None,
                'fraud_type': fraud_type,
                'financial_impact': invoice['Amt'] if expected_exception != 'NONE' else 0,
                'should_match': pattern == 'CLEAN'
            })
            
            invoice_counter += 1
        
        # Save dataset
        dataset_name = 'dataset1_corporate'
        pd.DataFrame(pos).to_csv(f'{self.output_dir}/{dataset_name}_pos.csv', index=False)
        pd.DataFrame(grs).to_csv(f'{self.output_dir}/{dataset_name}_grs.csv', index=False)
        pd.DataFrame(invoices).to_csv(f'{self.output_dir}/{dataset_name}_invoices.csv', index=False)
        pd.DataFrame(ground_truth).to_csv(f'{self.output_dir}/{dataset_name}_ground_truth.csv', index=False)
        
        print(f"Generated {len(invoices)} invoices, {len(pos)} POs, {len(grs)} GRs")
        return dataset_name
    
    def generate_dataset_2_legacy(self):
        """Dataset 2: Legacy ERP Format (abbreviated columns)"""
        print("\n=== Generating Dataset 2: Legacy ERP ===")
        
        pos = []
        grs = []
        invoices = []
        ground_truth = []
        
        invoice_counter = 1001  # Different range
        
        # Vendor collusion pattern - 3 vendors with coordinated fraud
        collusion_vendors = random.sample(self.vendors, 3)
        collusion_date = self.generate_date_range(60, 45)
        
        for i in range(100):
            # Timing-based fraud: 15 invoices clustered at quarter-end
            if 70 <= i < 85:
                pattern = 'CLEAN' if i % 3 == 0 else 'PRICE_VARIANCE'
                fraud_type = 'TIMING_EXPLOITATION' if pattern == 'PRICE_VARIANCE' else None
                # Quarter-end clustering
                po_date = datetime(2024, 12, random.randint(28, 31))  # End of Q4
            elif i < 60:
                pattern = 'CLEAN'
                fraud_type = None
                po_date = self.generate_date_range(180, 30)
            elif i < 68:  # Vendor collusion
                pattern = 'PRICE_VARIANCE'
                fraud_type = 'VENDOR_COLLUSION'
                po_date = collusion_date + timedelta(days=random.randint(0, 3))
            elif i < 75:
                pattern = 'NO_PO'
                fraud_type = None
                po_date = self.generate_date_range(90, 20)
            elif i < 82:
                pattern = 'QTY_OVERBILLING'
                fraud_type = 'PROGRESSIVE_OVERBILLING'
                po_date = self.generate_date_range(180, 30)
            elif i < 88:
                pattern = 'DUPLICATE'
                fraud_type = None
                po_date = self.generate_date_range(90, 20)
            else:
                pattern = 'NO_GR'
                fraud_type = None
                po_date = self.generate_date_range(90, 20)
            
            # Vendor selection
            if fraud_type == 'VENDOR_COLLUSION':
                vendor = collusion_vendors[i % 3]
            else:
                vendor = random.choice(self.vendors)
            
            po_id = f'PO{str(2000 + i).zfill(6)}'
            qty = random.randint(10, 500)
            
            # Progressive overbilling: gradually increase prices
            if fraud_type == 'PROGRESSIVE_OVERBILLING':
                base_price = random.uniform(100, 3000)
                multiplier = 1 + (i - 75) * 0.05  # 5% increase per invoice
                unit_price = round(base_price * multiplier, 2)
            else:
                unit_price = round(random.uniform(100, 5000), 2)
            
            po = {
                'po_ref': po_id,
                'supp_cd': vendor['id'],
                'supp_name': vendor['name'],
                'itm_cd': random.choice(self.items),
                'units': qty,
                'rate': unit_price,
                'dt': po_date.strftime('%Y-%m-%d'),
                'appr': random.choice(self.approvers),
                'sts': 'A',
                'net_amt': round(qty * unit_price, 2)
            }
            
            # Generate GR
            if pattern not in ['NO_GR', 'NO_PO']:
                gr_date = po_date + timedelta(days=random.randint(1, 25))
                received_qty = qty
                
                gr = {
                    'gr_ref': f'GR{str(2000 + len(grs)).zfill(6)}',
                    'po_ref': po_id,
                    'itm_cd': po['itm_cd'],
                    'recv_units': received_qty,
                    'recv_dt': gr_date.strftime('%Y-%m-%d'),
                    'recv_by': random.choice(self.receivers),
                    'supp_cd': vendor['id'],
                    'sts': 'C'
                }
                grs.append(gr)
                invoice_date = gr_date + timedelta(days=random.randint(1, 15))
            else:
                gr_date = po_date + timedelta(days=random.randint(5, 20))
                invoice_date = gr_date
                received_qty = qty
            
            # Generate Invoice
            if pattern == 'NO_PO':
                invoice_po = f'PO{str(9100 + i).zfill(6)}'
            else:
                invoice_po = po_id
                pos.append(po)
            
            if pattern == 'QTY_OVERBILLING':
                invoice_qty = received_qty + random.randint(15, 60)
            else:
                invoice_qty = received_qty
            
            if pattern == 'PRICE_VARIANCE':
                invoice_price = round(unit_price * random.uniform(1.13, 1.28), 2)
            else:
                invoice_price = unit_price
            
            invoice = {
                'inv_id': f'INV{str(invoice_counter).zfill(6)}',
                'supp_cd': vendor['id'],
                'supp_name': vendor['name'],
                'po_ref': invoice_po,
                'itm_cd': po['itm_cd'],
                'inv_units': invoice_qty,
                'inv_rate': invoice_price,
                'dt': invoice_date.strftime('%Y-%m-%d'),
                'net_amt': round(invoice_qty * invoice_price, 2),
                'sts': 'P'
            }
            invoices.append(invoice)
            
            # Duplicate handling
            if pattern == 'DUPLICATE':
                duplicate = invoice.copy()
                invoice_counter += 1
                duplicate['inv_id'] = f'INV{str(invoice_counter).zfill(6)}'
                duplicate['inv_units'] = invoice_qty + random.randint(-3, 3)
                duplicate['inv_rate'] = round(invoice_price * random.uniform(0.98, 1.02), 2)
                duplicate['net_amt'] = round(duplicate['inv_units'] * duplicate['inv_rate'], 2)
                duplicate['dt'] = (datetime.strptime(invoice['dt'], '%Y-%m-%d') + 
                                  timedelta(days=random.randint(-3, 4))).strftime('%Y-%m-%d')
                invoices.append(duplicate)
                
                ground_truth.append({
                    'invoice_id': duplicate['inv_id'],
                    'expected_exception': 'DUPLICATE',
                    'is_fraud': False,
                    'fraud_type': None,
                    'financial_impact': duplicate['net_amt'],
                    'should_match': False
                })
            
            expected_exception = 'NONE' if pattern == 'CLEAN' else pattern
            ground_truth.append({
                'invoice_id': invoice['inv_id'],
                'expected_exception': expected_exception,
                'is_fraud': fraud_type is not None,
                'fraud_type': fraud_type,
                'financial_impact': invoice['net_amt'] if expected_exception != 'NONE' else 0,
                'should_match': pattern == 'CLEAN'
            })
            
            invoice_counter += 1
        
        dataset_name = 'dataset2_legacy'
        pd.DataFrame(pos).to_csv(f'{self.output_dir}/{dataset_name}_pos.csv', index=False)
        pd.DataFrame(grs).to_csv(f'{self.output_dir}/{dataset_name}_grs.csv', index=False)
        pd.DataFrame(invoices).to_csv(f'{self.output_dir}/{dataset_name}_invoices.csv', index=False)
        pd.DataFrame(ground_truth).to_csv(f'{self.output_dir}/{dataset_name}_ground_truth.csv', index=False)
        
        print(f"Generated {len(invoices)} invoices, {len(pos)} POs, {len(grs)} GRs")
        return dataset_name
    
    def generate_dataset_3_international(self):
        """Dataset 3: International Format with invoice splitting"""
        print("\n=== Generating Dataset 3: International ===")
        
        pos = []
        grs = []
        invoices = []
        ground_truth = []
        
        invoice_counter = 2001
        
        for i in range(100):
            # Invoice splitting pattern: 10 cases where one PO is split into multiple invoices
            if 65 <= i < 75:
                pattern = 'CLEAN' if i == 65 else 'INVOICE_SPLIT'
                fraud_type = 'INVOICE_SPLITTING' if pattern == 'INVOICE_SPLIT' else None
                # Reference the same PO for splits
                if i > 65:
                    base_po_idx = 65
                else:
                    base_po_idx = i
            elif i < 60:
                pattern = 'CLEAN'
                fraud_type = None
                base_po_idx = i
            elif i < 65:
                pattern = 'NO_PO'
                fraud_type = 'SHELL_COMPANY'
                base_po_idx = i
            elif i < 80:
                pattern = 'PRICE_VARIANCE'
                fraud_type = 'ROUND_DOLLAR' if i % 2 == 0 else None
                base_po_idx = i
            elif i < 88:
                pattern = 'QTY_OVERBILLING'
                fraud_type = None
                base_po_idx = i
            elif i < 94:
                pattern = 'DUPLICATE'
                fraud_type = None
                base_po_idx = i
            else:
                pattern = 'NO_GR'
                fraud_type = None
                base_po_idx = i
            
            vendor = random.choice([v for v in self.vendors if v['is_shell']]) if fraud_type == 'SHELL_COMPANY' else random.choice(self.vendors)
            po_date = self.generate_date_range(180, 30)
            po_id = f'PO{str(3000 + base_po_idx).zfill(6)}'
            
            qty = random.randint(50, 500)
            
            if fraud_type == 'ROUND_DOLLAR':
                unit_price = random.choice([500, 1000, 2500, 5000])
            else:
                unit_price = round(random.uniform(100, 5000), 2)
            
            # Only create new PO if not splitting
            if pattern != 'INVOICE_SPLIT':
                po = {
                    'purchase_order': po_id,
                    'supplier_id': vendor['id'],
                    'supplier_name': vendor['name'],
                    'item_code': random.choice(self.items),
                    'quantity_ordered': qty,
                    'unit_cost': unit_price,
                    'order_date': po_date.strftime('%Y-%m-%d'),
                    'approved_by': random.choice(self.approvers),
                    'order_status': 'Approved',
                    'total_value': round(qty * unit_price, 2)
                }
                pos.append(po)
            else:
                # Reference existing PO
                po = [p for p in pos if p['purchase_order'] == po_id][0]
                # Split quantity among multiple invoices
                qty = po['quantity_ordered'] // (75 - 65)  # Divide among splits
                unit_price = po['unit_cost']
            
            # Generate GR
            if pattern not in ['NO_GR', 'NO_PO'] and pattern != 'INVOICE_SPLIT':
                gr_date = po_date + timedelta(days=random.randint(1, 25))
                received_qty = qty
                
                gr = {
                    'goods_receipt': f'GR{str(3000 + len(grs)).zfill(6)}',
                    'purchase_order': po_id,
                    'item_code': po['item_code'],
                    'quantity_received': received_qty,
                    'receipt_date': gr_date.strftime('%Y-%m-%d'),
                    'received_by': random.choice(self.receivers),
                    'supplier_id': vendor['id'],
                    'receipt_status': 'Completed'
                }
                grs.append(gr)
                invoice_date = gr_date + timedelta(days=random.randint(1, 15))
            elif pattern == 'INVOICE_SPLIT':
                # Use existing GR
                gr = [g for g in grs if g['purchase_order'] == po_id][0]
                received_qty = qty
                invoice_date = datetime.strptime(gr['receipt_date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 10))
                invoice_date = invoice_date.strftime('%Y-%m-%d')
            else:
                gr_date = po_date + timedelta(days=random.randint(5, 20))
                invoice_date = gr_date.strftime('%Y-%m-%d')
                received_qty = qty
            
            # Generate Invoice
            if pattern == 'NO_PO':
                invoice_po = f'PO{str(9200 + i).zfill(6)}'
            else:
                invoice_po = po_id
            
            if pattern == 'QTY_OVERBILLING':
                invoice_qty = received_qty + random.randint(20, 70)
            else:
                invoice_qty = received_qty
            
            if pattern == 'PRICE_VARIANCE':
                invoice_price = round(unit_price * random.uniform(1.14, 1.30), 2)
            else:
                invoice_price = unit_price
            
            invoice = {
                'invoice_number': f'INV{str(invoice_counter).zfill(6)}',
                'supplier_id': vendor['id'],
                'supplier_name': vendor['name'],
                'purchase_order': invoice_po,
                'item_code': po['item_code'] if pattern != 'NO_PO' else random.choice(self.items),
                'quantity_invoiced': invoice_qty,
                'unit_price': invoice_price,
                'invoice_dt': invoice_date if isinstance(invoice_date, str) else invoice_date.strftime('%Y-%m-%d'),
                'total_value': round(invoice_qty * invoice_price, 2),
                'payment_status': 'Pending'
            }
            invoices.append(invoice)
            
            # Duplicate handling
            if pattern == 'DUPLICATE':
                duplicate = invoice.copy()
                invoice_counter += 1
                duplicate['invoice_number'] = f'INV{str(invoice_counter).zfill(6)}'
                duplicate['quantity_invoiced'] = invoice_qty + random.randint(-4, 4)
                duplicate['unit_price'] = round(invoice_price * random.uniform(0.97, 1.03), 2)
                duplicate['total_value'] = round(duplicate['quantity_invoiced'] * duplicate['unit_price'], 2)
                base_date = invoice['invoice_dt'] if isinstance(invoice['invoice_dt'], str) else invoice['invoice_dt'].strftime('%Y-%m-%d')
                duplicate['invoice_dt'] = (datetime.strptime(base_date, '%Y-%m-%d') + 
                                           timedelta(days=random.randint(-4, 5))).strftime('%Y-%m-%d')
                invoices.append(duplicate)
                
                ground_truth.append({
                    'invoice_id': duplicate['invoice_number'],
                    'expected_exception': 'DUPLICATE',
                    'is_fraud': False,
                    'fraud_type': None,
                    'financial_impact': duplicate['total_value'],
                    'should_match': False
                })
            
            # Special handling for invoice splits - they should match but flagged for review
            if pattern == 'INVOICE_SPLIT':
                expected_exception = 'NONE'  # They technically match, just split
                should_match = True
            else:
                expected_exception = 'NONE' if pattern == 'CLEAN' else pattern
                should_match = pattern == 'CLEAN'
            
            ground_truth.append({
                'invoice_id': invoice['invoice_number'],
                'expected_exception': expected_exception,
                'is_fraud': fraud_type is not None,
                'fraud_type': fraud_type,
                'financial_impact': invoice['total_value'] if expected_exception != 'NONE' else 0,
                'should_match': should_match
            })
            
            invoice_counter += 1
        
        dataset_name = 'dataset3_international'
        pd.DataFrame(pos).to_csv(f'{self.output_dir}/{dataset_name}_pos.csv', index=False)
        pd.DataFrame(grs).to_csv(f'{self.output_dir}/{dataset_name}_grs.csv', index=False)
        pd.DataFrame(invoices).to_csv(f'{self.output_dir}/{dataset_name}_invoices.csv', index=False)
        pd.DataFrame(ground_truth).to_csv(f'{self.output_dir}/{dataset_name}_ground_truth.csv', index=False)
        
        print(f"Generated {len(invoices)} invoices, {len(pos)} POs, {len(grs)} GRs")
        return dataset_name
    
    def generate_all_evaluation_data(self):
        """Generate all evaluation datasets"""
        print("="*60)
        print("EVALUATION DATA GENERATION")
        print("="*60)
        
        datasets = []
        datasets.append(self.generate_dataset_1_corporate())
        datasets.append(self.generate_dataset_2_legacy())
        datasets.append(self.generate_dataset_3_international())
        
        # Create summary
        summary = {
            'datasets': datasets,
            'total_invoices': 300,  # ~100 per dataset
            'fraud_patterns': [
                'SHELL_COMPANY',
                'ROUND_DOLLAR',
                'BELOW_THRESHOLD',
                'SYSTEMATIC_OVERBILLING',
                'TIMING_EXPLOITATION',
                'VENDOR_COLLUSION',
                'PROGRESSIVE_OVERBILLING',
                'INVOICE_SPLITTING'
            ],
            'exception_types': [
                'NO_PO',
                'NO_GR',
                'PRICE_VARIANCE',
                'QTY_OVERBILLING',
                'DUPLICATE'
            ]
        }
        
        with open(f'{self.output_dir}/summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n" + "="*60)
        print("GENERATION COMPLETE")
        print("="*60)
        print(f"\nTotal datasets: {len(datasets)}")
        print(f"Total invoices: ~300")
        print(f"Fraud patterns: {len(summary['fraud_patterns'])}")
        print(f"\nFiles saved in '{self.output_dir}/' directory")
        print("\nNext step: Run evaluation harness to test your system!")
        
        return summary

if __name__ == "__main__":
    generator = EvaluationDataGenerator()
    summary = generator.generate_all_evaluation_data()