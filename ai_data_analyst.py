import tempfile
import csv
import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import traceback
import json
import firebase_admin
from firebase_admin import credentials, firestore

# Import our custom executor utilities
from executor import PythonCodeExecutor, export_to_notebook, generate_code_with_llm

# Page config
st.set_page_config(
    page_title="DataMate AI - AI Data Analysis Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.5px;
    }
    
    /* Sleek metric cards */
    .eda-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #334155;
        margin-bottom: 12px;
    }
    
    .eda-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    
    .eda-label {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
    }
    
    /* Code output styling */
    .stCodeBlock {
        border-radius: 8px;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.json'):
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            return None, None
        
        # Ensure string columns are properly clean
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        return df, df.columns.tolist()
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None

# Helper to fetch DB Schema
def get_db_schema_summary(db_conn, db_type):
    summary = ""
    try:
        if db_type == "SQLite":
            cursor = db_conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            summary += f"Available Tables in SQLite Database: {tables}\n"
            for t in tables:
                cursor.execute(f"PRAGMA table_info({t});")
                cols = [f"{row[1]} ({row[2]})" for row in cursor.fetchall()]
                summary += f"  - Table '{t}' columns: {cols}\n"
        elif db_type in ["PostgreSQL", "MySQL"]:
            cursor = db_conn.cursor()
            if db_type == "PostgreSQL":
                cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            else:
                cursor.execute("SHOW TABLES;")
            tables = [row[0] for row in cursor.fetchall()]
            summary += f"Available Tables in Database: {tables}\n"
            for t in tables:
                if db_type == "PostgreSQL":
                    cursor.execute(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name='{t}';")
                else:
                    cursor.execute(f"DESCRIBE {t};")
                cols = [f"{row[0]} ({row[1]})" for row in cursor.fetchall()]
                summary += f"  - Table '{t}' columns: {cols}\n"
        elif db_type == "Firebase":
            # List root collections in firestore
            collections = [col.id for col in db_conn.collections()]
            summary += f"Available Collections in Firebase Firestore: {collections}\n"
    except Exception as e:
        summary += f"Could not fetch database schema details: {str(e)}\n"
    return summary

# Render Auto-EDA stats
def show_auto_eda(df):
    st.subheader("📊 Auto-EDA: Data Health Report")
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="eda-card"><div class="eda-value">' + f"{df.shape[0]:,}" + '</div><div class="eda-label">Total Rows</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="eda-card"><div class="eda-value">' + f"{df.shape[1]}" + '</div><div class="eda-label">Total Columns</div></div>', unsafe_allow_html=True)
    with col3:
        missing_count = df.isnull().sum().sum()
        total_cells = df.size
        missing_pct = (missing_count / total_cells) * 100 if total_cells > 0 else 0
        st.markdown('<div class="eda-card"><div class="eda-value">' + f"{missing_pct:.1f}%" + '</div><div class="eda-label">Missing Values</div></div>', unsafe_allow_html=True)
    with col4:
        dup_count = df.duplicated().sum()
        st.markdown('<div class="eda-card"><div class="eda-value">' + f"{dup_count:,}" + '</div><div class="eda-label">Duplicate Rows</div></div>', unsafe_allow_html=True)
        
    # Column Details Tab
    tab1, tab2 = st.tabs(["📋 Column Specifications", "📈 Summary Statistics"])
    with tab1:
        spec_data = []
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            spec_data.append({
                "Column Name": col,
                "Data Type": str(df[col].dtype),
                "Non-Null Count": len(df) - null_count,
                "Missing Percentage": f"{null_pct:.1f}%",
                "Unique Values": unique_count
            })
        st.dataframe(pd.DataFrame(spec_data), use_container_width=True)
        
    with tab2:
        # Numeric Summary
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty:
            st.write("**Numerical Columns Summary:**")
            st.dataframe(num_df.describe().T, use_container_width=True)
        # Categorical Summary
        cat_df = df.select_dtypes(exclude=[np.number])
        if not cat_df.empty:
            st.write("**Categorical Columns Summary:**")
            st.dataframe(cat_df.describe().T, use_container_width=True)

# System Message for code generation
SYSTEM_PROMPT = """You are DataMate AI, an autonomous data scientist agent.
Your goal is to answer the user's data question by writing and executing Python code.

You have access to:
- A pandas DataFrame named `df` (if uploaded).
- A database connection object named `db_conn` (if connected).

When the user asks you a question, you must:
1. Explain your plan briefly in 1-2 sentences.
2. Write a single python code block enclosed in ```python and ```.
3. IMPORTANT: When you want to display dataframes, print statements, or calculation results, you MUST use the print() function. For example, use print(df.head()) or print(accuracy). If you want to show a plot, simply build it using matplotlib/seaborn and DO NOT call plt.show(); our execution engine will automatically capture the active figure.
4. Keep the code self-contained and correct. Do not refer to variables that don't exist.
5. If you need to train a model, use scikit-learn.

Wait, if you need to fetch data from the database, you can write python code that queries the database using `db_conn` (e.g., `pd.read_sql_query(query, db_conn)`).

Example response format:
To find the sales total:
```python
total_sales = df['sales'].sum()
print(f"Total Sales: ${total_sales:,.2f}")
```
"""

# Extract code block from text
def extract_code(text):
    import re
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    pattern_generic = r"```(.*?)```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1).strip()
        
    return text.strip()

# Run the Self-Correction Agent Loop
def run_self_correction_loop(user_query, provider, api_key, model_name, df, db_conn, status_placeholder):
    executor = PythonCodeExecutor()
    
    # Construct schema info
    schema_info = ""
    if df is not None:
        schema_info += f"DataFrame Info:\n- Shape: {df.shape[0]} rows, {df.shape[1]} columns\n"
        schema_info += "- Columns, Non-Null Count, Dtype:\n"
        for col in df.columns:
            non_null = df[col].notnull().sum()
            dtype = df[col].dtype
            schema_info += f"  * '{col}': {non_null} non-null values, type {dtype}\n"
        schema_info += f"\nSample Data (first 3 rows):\n{df.head(3).to_string()}\n"
        
    if db_conn is not None:
        schema_info += "\nDatabase connection is active. Use pd.read_sql_query(query, db_conn) for tables.\n"

    messages = [
        {"role": "user", "content": f"{schema_info}\n\nUser Request: {user_query}"}
    ]
    
    max_retries = 3
    executed_codes = []
    
    for attempt in range(max_retries):
        status_placeholder.write(f"🤖 Generating analysis code (Attempt {attempt + 1}/{max_retries})...")
        try:
            response = generate_code_with_llm(
                provider=provider,
                api_key=api_key,
                model_name=model_name,
                messages=messages,
                system_message=SYSTEM_PROMPT
            )
        except Exception as llm_err:
            return {
                "success": False,
                "error": f"LLM Generation Error: {str(llm_err)}",
                "executed_codes": executed_codes,
                "plots": [],
                "output": ""
            }
            
        code = extract_code(response)
        executed_codes.append(code)
        
        status_placeholder.write(f"⚙️ Executing Python code...")
        result = executor.execute(code, df=df, db_conn=db_conn)
        
        if result["success"]:
            status_placeholder.write("✅ Execution successful!")
            return {
                "success": True,
                "output": result["stdout"],
                "plots": result["plots"],
                "explanation": response,
                "executed_codes": executed_codes,
                "stderr": result["stderr"]
            }
        else:
            status_placeholder.write(f"⚠️ Attempt {attempt + 1} failed. Self-correcting...")
            # Append history
            messages.append({"role": "assistant", "content": response})
            error_feedback = (
                f"The code execution failed. Traceback/Stderr:\n"
                f"```text\n{result['stderr']}\n```\n"
                f"Please inspect the error, adjust your imports or logic, and write corrected Python code inside a new ```python ... ``` block."
            )
            messages.append({"role": "user", "content": error_feedback})
            
    return {
        "success": False,
        "error": "Self-correction limit exceeded. The agent could not generate working code.",
        "executed_codes": executed_codes,
        "plots": [],
        "output": "",
        "stderr": result.get("stderr", "")
    }

# ----------------- UI Rendering -----------------

st.title("📊 DataMate AI - Junior Data Scientist")
st.markdown("##### Upload data or connect a database to ask questions, run analytics, build models, and export code.")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model Provider selection
    provider = st.selectbox("LLM Provider", ["Google Gemini", "OpenAI"])
    
    if provider == "Google Gemini":
        api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter Google Gemini API Key")
        model_name = st.text_input("Model ID", value="gemini-1.5-flash")
    else:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="Enter OpenAI API Key")
        model_name = st.text_input("Model ID", value="gpt-4o")

    # Database Connector Section
    st.markdown("---")
    st.subheader("🔌 Database Connection (Optional)")
    db_type = st.selectbox("Database Type", ["None", "SQLite", "PostgreSQL", "MySQL", "Firebase"])
    
    db_conn = None
    if db_type != "None":
        if db_type == "SQLite":
            sqlite_path = st.text_input("SQLite File Path", value="database.db")
            if st.button("🔌 Connect SQLite"):
                try:
                    db_conn = sqlite3.connect(sqlite_path, check_same_thread=False)
                    st.session_state.db_conn = db_conn
                    st.session_state.db_type = db_type
                    st.success("Connected to SQLite!")
                except Exception as e:
                    st.error(f"SQLite Connection Error: {e}")
        elif db_type == "Firebase":
            firebase_key = st.text_area("Firestore Service Account JSON or File Path", placeholder="Paste service account JSON or enter file path")
            collection_name = st.text_input("Firestore Collection Name", placeholder="e.g. users")
            if st.button("🔌 Connect Firebase"):
                try:
                    if not firebase_admin._apps:
                        if os.path.exists(firebase_key):
                            cred = credentials.Certificate(firebase_key)
                        else:
                            cred_dict = json.loads(firebase_key)
                            cred = credentials.Certificate(cred_dict)
                        firebase_admin.initialize_app(cred)
                    
                    db = firestore.client()
                    st.session_state.db_conn = db
                    st.session_state.db_type = db_type
                    st.success("Connected to Firebase Firestore!")
                    
                    if collection_name:
                        st.info(f"Fetching collection '{collection_name}'...")
                        docs = db.collection(collection_name).stream()
                        data = []
                        for doc in docs:
                            doc_dict = doc.to_dict()
                            doc_dict["_doc_id"] = doc.id
                            data.append(doc_dict)
                        if data:
                            st.session_state.firebase_df = pd.DataFrame(data)
                            st.success(f"Fetched {len(data)} documents from '{collection_name}'!")
                        else:
                            st.warning(f"Collection '{collection_name}' has no documents or does not exist.")
                except Exception as e:
                    st.error(f"Firebase connection error: {e}")
        else:
            host = st.text_input("Host", value="localhost")
            port = st.text_input("Port", value="5432" if db_type == "PostgreSQL" else "3306")
            user = st.text_input("Username")
            password = st.text_input("Password", type="password")
            dbname = st.text_input("Database Name")
            
            if st.button(f"🔌 Connect {db_type}"):
                try:
                    if db_type == "PostgreSQL":
                        import psycopg2
                        db_conn = psycopg2.connect(host=host, port=port, user=user, password=password, database=dbname)
                    else:
                        import pymysql
                        db_conn = pymysql.connect(host=host, port=port, user=user, password=password, database=dbname)
                    st.session_state.db_conn = db_conn
                    st.session_state.db_type = db_type
                    st.success(f"Connected to {db_type}!")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

    # Restore DB Connection from session state if active
    if "db_conn" in st.session_state:
        db_conn = st.session_state.db_conn
        db_type = st.session_state.db_type
        st.info(f"Connected to active {db_type} database")
        if st.button("❌ Disconnect Database"):
            try:
                if db_type != "Firebase":
                    st.session_state.db_conn.close()
                else:
                    for app in list(firebase_admin._apps.values()):
                        firebase_admin.delete_app(app)
            except:
                pass
            del st.session_state.db_conn
            del st.session_state.db_type
            if "firebase_df" in st.session_state:
                del st.session_state.firebase_df
            st.rerun();

    # Jupyter Notebook Export Button
    st.markdown("---")
    st.subheader("📓 Jupyter Handoff")
    
    # Collect successful codes from history
    executed_successful_codes = []
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            if msg["role"] == "assistant" and msg.get("success", False):
                if msg.get("codes"):
                    executed_successful_codes.append(msg["codes"][-1])
                    
    if executed_successful_codes:
        notebook_json = export_to_notebook(executed_successful_codes)
        st.download_button(
            label="📥 Export Notebook (.ipynb)",
            data=notebook_json,
            file_name="datamate_analysis.ipynb",
            mime="application/x-ipynb+json"
        )
    else:
        st.write("Complete successful queries to export code.")

# Main Panel layout
# File Upload Widget
uploaded_file = st.file_uploader("Upload CSV, Excel, or JSON", type=["csv", "xlsx", "xls", "json"])

df = None
if uploaded_file is not None:
    df, columns = preprocess_and_save(uploaded_file)
elif "firebase_df" in st.session_state:
    df = st.session_state.firebase_df

if df is not None:
    col_preview, col_eda = st.columns([2, 3])
    with col_preview:
        with st.expander("👀 Data Table Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
    with col_eda:
        with st.expander("📊 Auto-EDA Report", expanded=False):
            show_auto_eda(df)

# If database is connected, show schema info
if db_conn is not None:
    with st.expander("🔌 Database Tables & Schema Summary", expanded=True):
        st.text(get_db_schema_summary(db_conn, db_type))

# Chat History state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.write(msg["content"])
        else:
            # Assistant response
            st.markdown(msg.get("explanation", ""))
            
            # Show stdout prints if they exist
            if msg.get("output"):
                with st.expander("📟 Console Prints", expanded=True):
                    st.text(msg["output"])
                    
            # Show plots if they exist
            if msg.get("plots"):
                for p in msg["plots"]:
                    if os.path.exists(p):
                        st.image(p, use_container_width=True)
                        
            # Show codes executed
            if msg.get("codes"):
                with st.expander("💻 Code Executed", expanded=False):
                    for idx, code in enumerate(msg["codes"]):
                        st.text(f"Attempt {idx + 1}:")
                        st.code(code, language="python")

# Chat input field
user_input = st.chat_input("Ask DataMate AI about the dataset or model forecast...")

if user_input:
    # Check for API Key
    if not api_key:
        st.warning("Please provide your LLM API Key in the sidebar to run queries.")
    else:
        # Add user query to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            # Progress status block
            with st.status("DataMate AI is thinking...", expanded=True) as status_block:
                result = run_self_correction_loop(
                    user_query=user_input,
                    provider=provider,
                    api_key=api_key,
                    model_name=model_name,
                    df=df,
                    db_conn=db_conn,
                    status_placeholder=status_block
                )
                status_block.update(label="Analysis Completed!", state="complete")
                
            # Render response
            if result.get("success", False):
                st.markdown(result["explanation"])
                
                if result.get("output"):
                    with st.expander("📟 Console Prints", expanded=True):
                        st.text(result["output"])
                        
                if result.get("plots"):
                    for p in result["plots"]:
                        st.image(p, use_container_width=True)
                        
                if result.get("executed_codes"):
                    with st.expander("💻 Code Executed", expanded=False):
                        for idx, code in enumerate(result["executed_codes"]):
                            st.text(f"Attempt {idx + 1}:")
                            st.code(code, language="python")
                            
                # Save assistant output to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "explanation": result["explanation"],
                    "output": result["output"],
                    "plots": result["plots"],
                    "codes": result["executed_codes"],
                    "success": True
                })
            else:
                st.error(result.get("error", "An error occurred."))
                if result.get("stderr"):
                    st.code(result["stderr"], language="text")
                if result.get("executed_codes"):
                    with st.expander("💻 Failed Code Attempts", expanded=True):
                        for idx, code in enumerate(result["executed_codes"]):
                            st.text(f"Attempt {idx + 1}:")
                            st.code(code, language="python")
                            
                st.session_state.messages.append({
                    "role": "assistant",
                    "explanation": f"Failed analysis: {result.get('error')}",
                    "output": "",
                    "plots": [],
                    "codes": result.get("executed_codes", []),
                    "success": False
                })
            st.rerun()