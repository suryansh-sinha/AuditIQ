# 🧠 **AuditIQ**

**AuditIQ** is an **AI-powered multi-agent auditing system** that automates **three-way matching**, **exception detection**, and **risk analysis** in Accounts Payable operations.  
Built with **AutoGen**, **Streamlit**, and **LLM-based reasoning**, it provides intelligent, transparent, and auditable insights across invoices, purchase orders, and goods receipts.

---

## 🚀 **Overview**

Organizations often struggle with manual invoice verification, data inconsistencies, and fraud risks.  
AuditIQ automates this process through a collaborative network of AI agents, each specializing in a specific financial audit task.

It supports two modes:
- **Default Mode** → Fully automated audit workflow  
- **Query Mode** → Interactive data exploration through natural language queries  

---

## ⚡️ **Installation**

```bash
# Clone repository
git clone https://github.com/<your-username>/AuditIQ.git
cd AuditIQ

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GEMINI_API_KEY=your-api-key" > .env

# Launch app
streamlit run app.py
```

---

## ⚙️ **Core Workflow**

### 🧾 **Default Mode – Automated Audit**

```markdown
User uploads CSVs
↓
1️⃣ DataMapperAgent → Schema & key analysis
2️⃣ MatchingAgent → Invoice–PO–GR reconciliation
3️⃣ AnalysisAgent → Risk synthesis & fraud detection
4️⃣ ReportAgent → LLM-generated markdown report
↓
✅ User receives full audit report with risk insights
```

### 🔍 **Query Mode – Data Exploration**

```markdown
User enters Query Mode
↓
QueryAgent interprets natural language queries
↓
Returns filtered results, explanations, and generated Python code
```

---

## 🧠 Agents

| Agent               | Role                 | Core Responsibilities                                                 |
| ------------------- | -------------------- | --------------------------------------------------------------------- |
| **DataMapperAgent** | Schema Understanding | Detects and aligns columns, identifies data quality issues            |
| **MatchingAgent**   | Three-Way Matching   | Matches Invoices ↔ POs ↔ GRs and flags exceptions                     |
| **AnalysisAgent**   | Risk Analysis        | Identifies vendor anomalies, fraud indicators, and exception patterns |
| **ReportAgent**     | Report Generation    | Uses an LLM to produce professional markdown audit reports            |
| **QueryAgent**      | Interactive Querying | Translates natural language queries into pandas code for analysis     |

---

## 💡 Key Features

- 🔗 Automated 3-Way Matching — Seamlessly reconcile invoices, POs, and GRs
- ⚠️ Exception Detection — Detects missing POs, quantity mismatches, duplicates, and overbilling
- 🧩 Schema Intelligence — Automatically maps CSVs of varying formats
- 📊 Risk Assessment — Scores vendor risk and identifies high-exposure patterns
- 🧠 LLM-Generated Reports — Produces human-readable markdown reports
- 💬 Natural Language Querying — “Show invoices without purchase orders”
- 🧾 Full Audit Trail — Transparent, reproducible, and explainable

---

## 🧑‍💻 **Sample Natural Language Queries**

| Example Query                                      | Agent Involved             | Description                            |
| -------------------------------------------------- | -------------------------- | -------------------------------------- |
| “Find all invoices without purchase orders”        | MatchingAgent + QueryAgent | Detects invoices missing PO references |
| “Show me price variances above 15%”                | MatchingAgent              | Flags mismatched pricing               |
| “Which vendors have the most exceptions?”          | AnalysisAgent              | Aggregates exceptions by vendor        |
| “List invoices with duplicate IDs”                 | MatchingAgent              | Identifies duplicates                  |
| “What is the financial exposure from overbilling?” | AnalysisAgent              | Calculates total overpayment risk      |
| “Generate an executive summary report”             | ReportAgent                | Produces LLM-based markdown report     |

---

## 🖥️ **User Interface Flow**

```text
1️⃣ Launch Streamlit app
2️⃣ Upload invoice, PO, and GR CSV files
3️⃣ Choose:
   - Automated Audit Mode → Run full analysis
   - Query Mode → Ask data-driven questions
4️⃣ Review:
   - Exception tables
   - Vendor risk scores
   - Generated audit report (Markdown or PDF)
5️⃣ Download report or share via dashboard
```

---

## 📚 **Tech Stack**

- 🧩 AutoGen v0.7.5 — Multi-agent orchestration framework
- 🤖 Gemini 2.5 Flash — Natural language reasoning & report generation
- 🧠 Pandas / NumPy — Data manipulation
- 🎨 Streamlit — Interactive dashboard
- 🧱 Pydantic / Dataclasses — Data model validation

---

## 📄 **License**

Released under the MIT License.
You are free to use, modify, and distribute with attribution.

---

## 🤝 **Contributing**

Contributions are welcome!
- Fork the repo
- Create a feature branch
- Submit a PR with a clear description

---

## 🧭 **Future Enhancements**

- Automated PDF export of reports
- Historical audit trend analysis
- Real-time vendor anomaly tracking
- Support for SAP / Oracle ERP data formats