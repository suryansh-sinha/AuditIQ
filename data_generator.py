"""
Data Generator for Three-Way Matching System
Generates synthetic Purchase Orders, Goods Receipts, and Invoices with controlled mismatches
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

class InvoiceDataGenerator:
    def __init__(self, output_dir='data'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate vendor pool
        self.vendors = [
            {'id': f'V{str(i).zfill(4)}', 'name': fake.company()}
            for i in range(1, 51)
        ]
        
        # Generate item codes
        self.items = (
            [f'IT-{str(i).zfill(3)}' for i in range(1, 101)] +
            [f'RAW-{str(i).zfill(3)}' for i in range(1, 51)]
        )
        
        # Approvers and receivers
        self.approvers = [fake.name() for _ in range(10)]
        self.receivers = [fake.name() for _ in range(15)]
        
    def generate_date_range(self, start_days_ago=180, end_days_ago=0):
        """Generate random date within specified range"""
        start_date = datetime.now() - timedelta(days=start_days_ago)
        end_date = datetime.now() - timedelta(days=end_days_ago)
        random_days = random.randint(0, (end_date - start_date).days)
        return start_date + timedelta(days=random_days)
    
    def generate_purchase_orders(self, count=500):
        """Generate Purchase Orders"""
        pos = []
        
        for i in range(1, count + 1):
            vendor = random.choice(self.vendors)
            po_date = self.generate_date_range(180, 30)
            
            po = {
                'PO_ID': f'PO{str(i).zfill(6)}',
                'Vendor_ID': vendor['id'],
                'Vendor_Name': vendor['name'],
                'Item_Code': random.choice(self.items),
                'Quantity': random.randint(10, 500),
                'Unit_Price': round(random.uniform(100, 50000), 2),
                'PO_Date': po_date.strftime('%Y-%m-%d'),
                'Approver': random.choice(self.approvers),
                'Status': 'Approved',
                'Total_Amount': 0  # Will calculate
            }
            po['Total_Amount'] = round(po['Quantity'] * po['Unit_Price'], 2)
            pos.append(po)
        
        return pd.DataFrame(pos)
    
    def generate_goods_receipts(self, pos_df, count=450):
        """Generate Goods Receipts linked to POs"""
        grs = []
        
        # Select subset of POs that have goods receipts
        selected_pos = pos_df.sample(n=min(count, len(pos_df)), replace=False)
        
        for idx, po in selected_pos.iterrows():
            po_date = datetime.strptime(po['PO_Date'], '%Y-%m-%d')
            gr_date = po_date + timedelta(days=random.randint(1, 30))
            
            # 95% of the time, receive exact quantity; 5% partial receipt
            if random.random() < 0.95:
                received_qty = po['Quantity']
            else:
                received_qty = random.randint(int(po['Quantity'] * 0.7), po['Quantity'] - 1)
            
            gr = {
                'GR_ID': f'GR{str(len(grs) + 1).zfill(6)}',
                'PO_ID': po['PO_ID'],
                'Item_Code': po['Item_Code'],
                'Received_Qty': received_qty,
                'GR_Date': gr_date.strftime('%Y-%m-%d'),
                'Receiver': random.choice(self.receivers),
                'Vendor_ID': po['Vendor_ID'],
                'Status': 'Completed'
            }
            grs.append(gr)
        
        return pd.DataFrame(grs)
    
    def generate_invoices_with_mismatches(self, pos_df, grs_df):
        """Generate Invoices with controlled mismatches"""
        invoices = []
        invoice_counter = 1
        
        # Get POs with GRs for valid matches
        pos_with_gr = grs_df['PO_ID'].unique()
        valid_pos = pos_df[pos_df['PO_ID'].isin(pos_with_gr)]
        
        # 1. Generate 380 perfect matches
        perfect_matches = valid_pos.sample(n=min(380, len(valid_pos)), replace=False)
        for idx, po in perfect_matches.iterrows():
            gr = grs_df[grs_df['PO_ID'] == po['PO_ID']].iloc[0]
            invoice_date = datetime.strptime(gr['GR_Date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 15))
            
            invoice = {
                'Invoice_ID': f'INV{str(invoice_counter).zfill(6)}',
                'Vendor_ID': po['Vendor_ID'],
                'Vendor_Name': po['Vendor_Name'],
                'PO_ID': po['PO_ID'],
                'Item_Code': po['Item_Code'],
                'Invoice_Qty': gr['Received_Qty'],
                'Invoice_Price': po['Unit_Price'],
                'Invoice_Date': invoice_date.strftime('%Y-%m-%d'),
                'Invoice_Amount': round(gr['Received_Qty'] * po['Unit_Price'], 2),
                'Status': 'Pending',
                'Mismatch_Type': 'None'
            }
            invoices.append(invoice)
            invoice_counter += 1
        
        # 2. Generate 30 invoices without matching PO (fraud indicator)
        for _ in range(30):
            vendor = random.choice(self.vendors)
            qty = random.randint(10, 200)
            price = round(random.uniform(500, 10000), 2)
            
            invoice = {
                'Invoice_ID': f'INV{str(invoice_counter).zfill(6)}',
                'Vendor_ID': vendor['id'],
                'Vendor_Name': vendor['name'],
                'PO_ID': f'PO{str(random.randint(9000, 9999)).zfill(6)}',  # Non-existent PO
                'Item_Code': random.choice(self.items),
                'Invoice_Qty': qty,
                'Invoice_Price': price,
                'Invoice_Date': self.generate_date_range(30, 0).strftime('%Y-%m-%d'),
                'Invoice_Amount': round(qty * price, 2),
                'Status': 'Pending',
                'Mismatch_Type': 'No_PO'
            }
            invoices.append(invoice)
            invoice_counter += 1
        
        # 3. Generate 25 invoices where Invoice_Qty > Received_Qty (overbilling)
        overbilling_pos = valid_pos.sample(n=min(25, len(valid_pos)), replace=True)
        for idx, po in overbilling_pos.iterrows():
            gr = grs_df[grs_df['PO_ID'] == po['PO_ID']].iloc[0]
            invoice_qty = gr['Received_Qty'] + random.randint(5, 50)
            invoice_date = datetime.strptime(gr['GR_Date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 15))
            
            invoice = {
                'Invoice_ID': f'INV{str(invoice_counter).zfill(6)}',
                'Vendor_ID': po['Vendor_ID'],
                'Vendor_Name': po['Vendor_Name'],
                'PO_ID': po['PO_ID'],
                'Item_Code': po['Item_Code'],
                'Invoice_Qty': invoice_qty,
                'Invoice_Price': po['Unit_Price'],
                'Invoice_Date': invoice_date.strftime('%Y-%m-%d'),
                'Invoice_Amount': round(invoice_qty * po['Unit_Price'], 2),
                'Status': 'Pending',
                'Mismatch_Type': 'Qty_Overbilling'
            }
            invoices.append(invoice)
            invoice_counter += 1
        
        # 4. Generate 20 invoices where Invoice_Price > PO Unit_Price by >10% (price variance)
        price_variance_pos = valid_pos.sample(n=min(20, len(valid_pos)), replace=True)
        for idx, po in price_variance_pos.iterrows():
            gr = grs_df[grs_df['PO_ID'] == po['PO_ID']].iloc[0]
            invoice_price = po['Unit_Price'] * random.uniform(1.11, 1.30)  # 11-30% higher
            invoice_date = datetime.strptime(gr['GR_Date'], '%Y-%m-%d') + timedelta(days=random.randint(1, 15))
            
            invoice = {
                'Invoice_ID': f'INV{str(invoice_counter).zfill(6)}',
                'Vendor_ID': po['Vendor_ID'],
                'Vendor_Name': po['Vendor_Name'],
                'PO_ID': po['PO_ID'],
                'Item_Code': po['Item_Code'],
                'Invoice_Qty': gr['Received_Qty'],
                'Invoice_Price': round(invoice_price, 2),
                'Invoice_Date': invoice_date.strftime('%Y-%m-%d'),
                'Invoice_Amount': round(gr['Received_Qty'] * invoice_price, 2),
                'Status': 'Pending',
                'Mismatch_Type': 'Price_Variance'
            }
            invoices.append(invoice)
            invoice_counter += 1
        
        # 5. Generate 15 duplicate invoices (same PO_ID, amount, vendor)
        perfect_matches = [inv for inv in invoices if inv['Mismatch_Type'] == 'None']
        duplicate_source = random.sample(perfect_matches, min(15, len(perfect_matches)))
        for source_invoice in duplicate_source:
            duplicate = source_invoice.copy()
            duplicate['Invoice_ID'] = f'INV{str(invoice_counter).zfill(6)}'
            
            # Introduce realistic variations
            qty_variation = random.randint(-2, 2)
            price_variation = random.uniform(0.99, 1.02)
            
            duplicate['Invoice_Qty'] = max(1, source_invoice['Invoice_Qty'] + qty_variation)
            duplicate['Invoice_Price'] = round(source_invoice['Invoice_Price'] * price_variation, 2)
            duplicate['Invoice_Amount'] = round(duplicate['Invoice_Qty'] * duplicate['Invoice_Price'], 2)

            # Vary date slightly
            duplicate['Invoice_Date'] = (
                datetime.strptime(source_invoice['Invoice_Date'], '%Y-%m-%d') + timedelta(days=random.randint(-3, 5))
            ).strftime('%Y-%m-%d')

            duplicate['Mismatch_Type'] = 'Duplicate'
            invoices.append(duplicate)
            invoice_counter += 1

        
        # 6. Generate 10 invoices without corresponding GR (payment before receipt)
        pos_without_gr = pos_df[~pos_df['PO_ID'].isin(pos_with_gr)].head(10)
        for idx, po in pos_without_gr.iterrows():
            invoice_date = datetime.strptime(po['PO_Date'], '%Y-%m-%d') + timedelta(days=random.randint(5, 20))
            
            invoice = {
                'Invoice_ID': f'INV{str(invoice_counter).zfill(6)}',
                'Vendor_ID': po['Vendor_ID'],
                'Vendor_Name': po['Vendor_Name'],
                'PO_ID': po['PO_ID'],
                'Item_Code': po['Item_Code'],
                'Invoice_Qty': po['Quantity'],
                'Invoice_Price': po['Unit_Price'],
                'Invoice_Date': invoice_date.strftime('%Y-%m-%d'),
                'Invoice_Amount': round(po['Quantity'] * po['Unit_Price'], 2),
                'Status': 'Pending',
                'Mismatch_Type': 'No_GR'
            }
            invoices.append(invoice)
            invoice_counter += 1
        
        return pd.DataFrame(invoices)
    
    def generate_all_data(self):
        """Generate all datasets and save to CSV"""
        print("Generating Purchase Orders...")
        pos_df = self.generate_purchase_orders(500)
        pos_df.to_csv(os.path.join(self.output_dir, 'purchase_orders.csv'), index=False)
        print(f"Generated {len(pos_df)} Purchase Orders")
        
        print("Generating Goods Receipts...")
        grs_df = self.generate_goods_receipts(pos_df, 450)
        grs_df.to_csv(os.path.join(self.output_dir, 'goods_receipts.csv'), index=False)
        print(f"Generated {len(grs_df)} Goods Receipts")
        
        print("Generating Invoices with controlled mismatches...")
        invoices_df = self.generate_invoices_with_mismatches(pos_df, grs_df)
        invoices_df.to_csv(os.path.join(self.output_dir, 'invoices.csv'), index=False)
        print(f"Generated {len(invoices_df)} Invoices")
        
        # Print mismatch summary
        print("\n=== Mismatch Summary ===")
        mismatch_summary = invoices_df['Mismatch_Type'].value_counts()
        print(mismatch_summary)
        
        return pos_df, grs_df, invoices_df

if __name__ == "__main__":
    generator = InvoiceDataGenerator()
    pos_df, grs_df, invoices_df = generator.generate_all_data()
    print("\nData generation complete! Files saved in 'data' directory.")