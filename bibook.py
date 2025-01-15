import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import sqlite3

# Load environment variables
load_dotenv()

# Configure Google Gemini API
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Initialize Gemini model
model = genai.GenerativeModel("gemini-1.5-flash")

def initialize_agents():
    """Initialize all three agents with specific roles"""
    agent1 = model.start_chat(
        history=[
            {"role": "user", "parts": "You are Agent 1, a data preprocessing specialist. Your role is to analyze visualization queries and create a refined, brief but comprehensive query that includes any necessary preprocessing steps and visualization requirements."},
            {"role": "model", "parts": "I understand my role as Agent 1. I will create clear, concise visualization queries that include all necessary details."}
        ]
    )
    
    agent2 = model.start_chat(
    history=[
        {"role": "user", "parts": """You are Agent 2, a plotly visualization specialist. You will take queries and generate appropriate plotly visualization code. 
        IMPORTANT RULES:
        - The dataframe 'df' already exists but filtered data doesnot exist
                
        """},
        {"role": "model", "parts": "I understand my role as Agent 2. I will create visualization code based on refined queries using the existing dataframe 'df', include all necessary preprocessing steps, and ensure proper code formatting and JSON-serializable data."}
        ]
    )

    agent3 = model.start_chat(
        history=[
            {"role": "user", "parts": "You are Agent 3, an error resolution specialist. Your role is to analyze error messages from visualization code execution and generate corrected code. Work only with the existing dataframe 'df'."},
            {"role": "model", "parts": "I understand my role as Agent 3. I will analyze errors and provide corrected visualization code that resolves the specific issues."}
        ]
    )
    
    return agent1, agent2, agent3

def ask_agent(message, agent):
    try:
        response = agent.send_message(message)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# def load_file(file):
#     try:
#         file_extension = file.name.split('.')[-1].lower()
#         if file_extension == 'csv':
#             df = pd.read_csv(file)
#         elif file_extension in ['xlsx', 'xls']:
#             df = pd.read_excel(file)
#         else:
#             return None, "Unsupported file format"
#         return df, None
#     except Exception as e:
#         return None, f"Error loading file: {str(e)}"



def load_file(file):
    try:
        file_extension = file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            df = pd.read_csv(file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file)
        elif file_extension in ['sqlite', 'db']:
            # Connect to the SQLite database
            conn = sqlite3.connect(file)
            # Fetch the list of tables in the database
            tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
            if tables.empty:
                return None, "SQLite file contains no tables"
            
            # Load the first table as a DataFrame
            first_table_name = tables['name'][0]
            df = pd.read_sql_query(f"SELECT * FROM {first_table_name};", conn)
            
            conn.close()
        else:
            return None, "Unsupported file format"
        
        return df, None
    except Exception as e:
        return None, f"Error loading file: {str(e)}"


def refine_query(query, df, agent1):
    """Agent 1: Analyzes and refines the visualization query."""
    prompt = f"""
    Given the following visualization request: '{query}'
    
    The existing dataframe 'df' has:
    Columns: {list(df.columns)}
    Data types: {df.dtypes.to_dict()}
        
    Provide a single refined query/ question in that:
    1. Is as brief as possible while being comprehensive
    2. Includes any necessary preprocessing steps on required columns only
    3. Specifies visualization type and requirements. If visualization type is not mentioned, suggest a suitable visualization.
    4. Uses only existing columns
    
    Format your response as JSON:
    {{"refined_query": "your brief but complete query here"}}
    
    Example: {{"refined_query": "Create a line chart of monthly sales (y-axis) over time (x-axis) after aggregating daily data. Use blue theme and include chart title."}}
    """
    
    response = ask_agent(prompt, agent1)
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            refined = json.loads(json_match.group())
        else:
            refined = json.loads(response)
        return refined.get("refined_query", query)
    except:
        st.error("Failed to parse Agent 1's response. Using original query.")
        return query

def generate_visualization_code(refined_query, df, agent2):
    """Agent 2: Generates visualization code based on refined query."""
    prompt = f"""
    Given the refined visualization request: "{refined_query}"
    
    The existing dataframe 'df' has:
    Columns: {list(df.columns)}
    Data types: {df.dtypes.to_dict()}
    Sample Data: {list(df.head(3))}
    
    Generate Python code for processing/ filtering and then plotly to create the visualization.
    IMPORTANT RULES:
    - dataframe 'df' already exists but filtered data doesnot exist
    - DO NOT include imports or try-except blocks
    - DO NOT include fig.show()
    - Always start code with df_copy = df and then proceed with further processing on df_copy
    
    
    """
    
    response = ask_agent(prompt, agent2)
    code = extract_code_from_response(response)
    return clean_code(code)

def extract_code_from_response(response):
    """Extract clean code from LLM response."""
    code = re.sub(r'```python\n?(.*?)\n?```', r'\1', response, flags=re.DOTALL)
    code = re.sub(r'```\n?(.*?)\n?```', r'\1', code, flags=re.DOTALL)
    code_lines = [line for line in code.split('\n') 
                 if not any(x in line.lower() for x in ['pd.dataframe', 'data =', 'df =', 'sample', '.csv'])]
    return '\n'.join(code_lines).strip()

def clean_code(code):
    """Clean up the generated code."""
    code = code.strip()
    code = re.sub(r'import.*?\n', '', code)
    code = re.sub(r'from.*?import.*?\n', '', code)
    code = re.sub(r'pd\.DataFrame\(.*?\)', '', code, flags=re.DOTALL)
    code = re.sub(r'df\s*=.*?\n', '', code)
    code = re.sub(r'print\(.*?\)\n?', '', code)
    return '\n'.join(line for line in code.split('\n') 
                    if line.strip() and not line.strip().startswith('#'))

def handle_error(error_message, original_code, df, agent3):
    """Agent 3: Analyzes error and generates corrected code."""
    prompt = f"""
    Error occurred executing:
    {original_code}
    
    Error message: {error_message}
    
    The dataframe 'df' has columns: {list(df.columns)}
    Data types: {df.dtypes.to_dict()}
    
    Provide corrected visualization code that resolves this error.
    Use ONLY the existing dataframe 'df'.
    """
    
    response = ask_agent(prompt, agent3)
    code = extract_code_from_response(response)
    return clean_code(code)


def execute_visualization_with_error_handling(df, code, agent3):
    """Execute visualization code with comprehensive error handling."""
    try:
        df_copy = df.copy()
              
        namespace = {
            'df': df_copy,
            'pd': pd,
            'np': np,
            'px': px,
            'go': go,
            'fig': None
        }
        
        # Execute the visualization code
        exec(code, namespace)
        
        if 'fig' not in namespace or namespace['fig'] is None:
            return None, "Code execution did not produce a figure", None
            
        # Try to serialize the figure to catch any remaining issues
        try:
            import plotly.io
            # Add validation to catch any problematic data
            plotly.io.to_json(namespace['fig'], validate=True)
        except Exception as json_error:
            st.error(f"JSON serialization error: {str(json_error)}")
            # Try to identify problematic data
            if hasattr(namespace['fig'], 'data'):
                for i, trace in enumerate(namespace['fig'].data):
                    try:
                        plotly.io.to_json(trace, validate=True)
                    except Exception as e:
                        st.error(f"Trace {i} contains non-serializable data: {str(e)}")
            return None, f"Figure contains non-serializable data: {str(json_error)}", None
            
        return namespace['fig'], None, None
            
    except Exception as e:
        st.error(f"Initial execution error: {str(e)}")
        corrected_code = handle_error(str(e), code, df, agent3)
        
        try:
            
            namespace = {
                'df': df_copy,
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'fig': None
            }
            
            exec(corrected_code, namespace)
            
            if 'fig' not in namespace or namespace['fig'] is None:
                return None, "Corrected code did not produce a figure", None
                
            try:
                plotly.io.to_json(namespace['fig'], validate=True)
            except Exception as json_error:
                return None, f"Corrected figure contains non-serializable data: {str(json_error)}", None
                
            return namespace['fig'], None, {'corrected_code': corrected_code}
            
        except Exception as e2:
            return None, f"Error in corrected code: {str(e2)}", None
            
        except Exception as e2:
            return None, f"Error in corrected code: {str(e2)}", None

def display_data_info(df):
    """Display organized data information."""
    st.markdown("### Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Rows", df.shape[0])
    with col2:
        st.metric("Total Columns", df.shape[1])
    
    with st.expander("Column Details", expanded=True):
        col_info = pd.DataFrame({
            'Data Type': df.dtypes,
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(col_info, use_container_width=True)
    
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head(5), use_container_width=True)

def format_code_block(code, language="python"):
    """Format code block with proper markdown."""
    return f"### 💻 Generated Code\n```{language}\n{code}\n```"

def main():
    st.set_page_config(
        page_title="Data Visualization Assistant",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "agent1" not in st.session_state or "agent2" not in st.session_state or "agent3" not in st.session_state:
        st.session_state.agent1, st.session_state.agent2, st.session_state.agent3 = initialize_agents()
    if "current_data" not in st.session_state:
        st.session_state.current_data = None
    if "input_key" not in st.session_state:
        st.session_state.input_key = 0

    st.title("BI BOOK: Data Visualization Assistant")
    
    with st.sidebar:
        st.header("Data Upload")
        uploaded_file = st.file_uploader(
            "Upload your data file (CSV or Excel)",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file:
            df, error = load_file(uploaded_file)
            if error:
                st.error(error)
            else:
                st.success("File uploaded successfully!")
                st.session_state.current_data = df
                st.write("### Data Info")
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
        if st.button("Clear History"):
            st.session_state.chat_history = []
            st.session_state.agent1, st.session_state.agent2, st.session_state.agent3 = initialize_agents()
            st.rerun()

    if st.session_state.current_data is not None:
        with st.expander("📋 Current Data", expanded=True):
            st.dataframe(st.session_state.current_data)

        for message in st.session_state.chat_history:
            with st.container():
                if message['role'] == "You":
                    st.markdown("#### 🔹 Your Query")
                    st.write(message['content'])
                else:
                    st.markdown("#### 🤖 Assistant Response")
                    if 'refined_query' in message:
                        st.markdown("### 🎯 Refined Query")
                        st.write(message['refined_query'])
                    if 'code' in message:
                        st.markdown(format_code_block(message['code']))
                    if 'error' in message:
                        st.error(message['error'])
                    if 'visualization' in message:
                        st.plotly_chart(message['visualization'], use_container_width=True)
                st.markdown("---")

        with st.container():
            st.markdown("### 🎨 Create Visualization")
            user_input = st.text_area(
                "Describe the visualization you want to create...",
                key=f"user_input_{st.session_state.input_key}",
                help="Describe what you want to visualize from your data"
            )

            if st.button("Generate Visualization", type="primary"):
                if user_input and user_input.strip():
                    st.session_state.chat_history.append({
                        "role": "You",
                        "content": user_input
                    })

                    try:
                        refined_query = refine_query(
                            user_input, 
                            st.session_state.current_data,
                            st.session_state.agent1
                        )
                        
                        generated_code = generate_visualization_code(
                            refined_query,
                            st.session_state.current_data,
                            st.session_state.agent2
                        )
                        
                        response_message = {
                            "role": "Assistant",
                            "refined_query": refined_query,
                            "code": generated_code
                        }

                        if not generated_code:
                            response_message["error"] = "Failed to generate visualization code"
                        else:
                            result, exec_error, error_info = execute_visualization_with_error_handling(
                                st.session_state.current_data, 
                                generated_code,
                                st.session_state.agent3
                            )

                            if exec_error:
                                if error_info:
                                    response_message["code"] = error_info['corrected_code']
                                response_message["error"] = exec_error
                            
                            if result is not None:
                                response_message["visualization"] = result
                        
                        st.session_state.chat_history.append(response_message)

                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "Assistant",
                            "error": f"An error occurred: {str(e)}"
                        })

                    st.session_state.input_key += 1
                    st.rerun()

    else:
        st.info("Please upload a data file to begin.")

if __name__ == "__main__":
    main()