import sys
import os
import io
import traceback
import contextlib
import pandas as pd
import numpy as np
import nbformat as nbf

class PythonCodeExecutor:
    def __init__(self, temp_plots_dir="temp_plots"):
        self.temp_plots_dir = temp_plots_dir
        os.makedirs(self.temp_plots_dir, exist_ok=True)
        
    def clear_plots(self):
        if os.path.exists(self.temp_plots_dir):
            for f in os.listdir(self.temp_plots_dir):
                if f.endswith('.png'):
                    try:
                        os.remove(os.path.join(self.temp_plots_dir, f))
                    except:
                        pass

    def execute(self, code_str: str, df: pd.DataFrame = None, db_conn = None):
        """
        Executes the code string in a local context.
        Injects df as 'df' and db_conn as 'db_conn'.
        Captures standard output and generated matplotlib plots.
        """
        # Save current matplotlib settings
        import matplotlib
        matplotlib.use('Agg')  # Safe for threads and non-GUI environments
        import matplotlib.pyplot as plt
        
        # Clear any existing plots in matplotlib state
        plt.close('all')
        
        # Clear plots folder to avoid confusion with previous runs
        self.clear_plots()

        # Set up redirection
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        # Build execution context
        globals_dict = {
            'pd': pd,
            'np': np,
            'plt': plt,
            'df': df,
            'db_conn': db_conn
        }
        
        # Include standard data science packages if available
        try:
            import seaborn as sns
            globals_dict['sns'] = sns
        except ImportError:
            pass
        try:
            import sklearn
            globals_dict['sklearn'] = sklearn
        except ImportError:
            pass
            
        success = True
        error_msg = ""
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                # Compile and execute the code
                exec(code_str, globals_dict)
            except Exception as e:
                success = False
                error_msg = traceback.format_exc()
        
        # Capture stdout and stderr
        captured_stdout = stdout_buf.getvalue()
        captured_stderr = stderr_buf.getvalue()
        
        # Capture plots
        plot_paths = []
        try:
            # Check if figures were created
            fig_nums = plt.get_fignums()
            if fig_nums:
                for idx, num in enumerate(fig_nums):
                    fig = plt.figure(num)
                    plot_filename = f"plot_{idx + 1}_{int(pd.Timestamp.now().timestamp())}.png"
                    plot_path = os.path.join(self.temp_plots_dir, plot_filename)
                    fig.savefig(plot_path, bbox_inches='tight', dpi=150)
                    plot_paths.append(plot_path)
                plt.close('all')
        except Exception as plot_err:
            captured_stderr += f"\nError capturing plots: {str(plot_err)}"
            
        return {
            "success": success,
            "stdout": captured_stdout,
            "stderr": captured_stderr if captured_stderr else (error_msg if not success else ""),
            "plots": plot_paths,
            "error_traceback": error_msg if not success else ""
        }

def export_to_notebook(code_blocks):
    """
    Given a list of code blocks (strings), generate a Jupyter Notebook JSON string.
    """
    nb = nbf.v4.new_notebook()
    nb.cells = []
    
    # Add a markdown title cell
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# DataMate AI - Analysis Notebook\n"
        "This notebook contains the code generated and executed by DataMate AI for your analysis."
    ))
    
    for idx, code in enumerate(code_blocks):
        nb.cells.append(nbf.v4.new_markdown_cell(f"## Cell {idx+1}"))
        nb.cells.append(nbf.v4.new_code_cell(code))
        
    return nbf.writes(nb)

def generate_code_with_llm(provider, api_key, model_name, messages, system_message):
    """
    Invokes the LLM to generate Python code.
    messages is a list of dicts: [{'role': 'user'/'assistant', 'content': '...'}]
    """
    if provider == "OpenAI":
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        
        api_messages = [{"role": "system", "content": system_message}]
        for msg in messages:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        response = client.chat.completions.create(
            model=model_name,
            messages=api_messages,
            temperature=0.1
        )
        return response.choices[0].message.content
        
    elif provider == "Google Gemini":
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Start model with system instruction
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_message
        )
        
        # Format chat history
        history = []
        for msg in messages[:-1]:
            # Gemini expects 'user' or 'model' roles
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [msg["content"]]})
        
        chat = model.start_chat(history=history)
        last_msg = messages[-1]["content"]
        response = chat.send_message(last_msg)
        return response.text
