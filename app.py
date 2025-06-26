import os
import sqlite3
import json 
from datetime import datetime
from flask import Flask, render_template, request, Response, stream_with_context, url_for 
from dotenv import load_dotenv
import openai
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Configure OpenAI API
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
else:
    print("Warning: OPENAI_API_KEY not found in .env file.")
    openai_client = None

# Configure Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model_name = 'gemini-2.5-pro' 
    gemini_model = genai.GenerativeModel(gemini_model_name)
else:
    print("Warning: GEMINI_API_KEY not found in .env file.")
    gemini_model = None

openai_model_name = 'gpt-4o-mini-2024-07-18' # For standard and creative loops
openai_paraphrase_model_name = 'gpt-4o-mini-2024-07-18' # Explicitly for paraphrasing
DB_NAME = 'logs.db'

# --- Flask App Initialization ---
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.secret_key = os.urandom(24)

# --- Database Helper Functions ---
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            ID INTEGER PRIMARY KEY AUTOINCREMENT,
            promptdate TEXT NOT NULL,
            prompt TEXT NOT NULL,
            result TEXT NOT NULL, 
            finalresult TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_log_to_db(user_prompt, openai_results_json, gemini_final_result_with_tokens):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute('''
            INSERT INTO logs (promptdate, prompt, result, finalresult)
            VALUES (?, ?, ?, ?)
        ''', (current_date, user_prompt, openai_results_json, gemini_final_result_with_tokens))
        conn.commit()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

# --- Routes ---
@app.route('/', methods=['GET'])
def index():
    if not os.path.exists(DB_NAME) or os.path.getsize(DB_NAME) == 0:
        init_db()
    else:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='logs';")
        if not cursor.fetchone():
            conn.close()
            init_db()
        else:
            conn.close()
    return render_template('index.html', initial_message="Enter a prompt below and click Go.")

@app.route('/overthink', methods=['GET']) 
def overthink():
    user_prompt = request.args.get('user_prompt', '').strip()

    def yield_sse_message(msg_type, content, additional_data=None):
        message_obj = {"type": msg_type, "content": content}
        if additional_data:
            message_obj.update(additional_data)
        return f"data: {json.dumps(message_obj)}\n\n"

    if not user_prompt:
        def error_stream_no_prompt():
            yield yield_sse_message("error", "Please enter a prompt.")
            yield f"event: stream_end\ndata: Error - No prompt\n\n"
        return Response(stream_with_context(error_stream_no_prompt()), mimetype='text/event-stream')

    if not openai_client or not gemini_model:
        def error_stream_config():
            yield yield_sse_message("error", "API client not configured on server. Please check API keys.")
            yield f"event: stream_end\ndata: Error - API config\n\n"
        return Response(stream_with_context(error_stream_config()), mimetype='text/event-stream')

    def generate_responses_stream(original_prompt):
        # ... (The first part of the function remains the same) ...
        yield yield_sse_message("status", f"Original Prompt: {original_prompt}")
        paraphrased_prompts_list = []
        openai_results_dict = {}
        total_openai_prompt_tokens = 0
        total_openai_completion_tokens = 0
        total_gemini_prompt_tokens = 0
        total_gemini_completion_tokens = 0
        
        # --- Paraphrasing Loop ---
        num_paraphrases = 15
        openai_system_prompt_paraphrase = "Rephrase the text in the prompt into a paraphrase to make it clearer. Note: Do not add nonexistent or remove existing things. Do not show your thinking, only return the paraphrased text."
        paraphrase_temperature = 0.7
        paraphrase_max_tokens = 1000
        yield yield_sse_message("status", "\n--- Generating Paraphrased Prompts ---")
        for k in range(num_paraphrases):
            iteration_text = f"Generating Paraphrase {k+1}/{num_paraphrases}..."
            yield yield_sse_message("status", iteration_text)
            try:
                response = openai_client.chat.completions.create(model=openai_paraphrase_model_name, messages=[{"role": "system", "content": openai_system_prompt_paraphrase}, {"role": "user", "content": original_prompt}], temperature=paraphrase_temperature, max_tokens=paraphrase_max_tokens)
                paraphrased_text = response.choices[0].message.content.strip()
                paraphrased_prompts_list.append(paraphrased_text)
                if response.usage:
                    total_openai_prompt_tokens += response.usage.prompt_tokens
                    total_openai_completion_tokens += response.usage.completion_tokens
                yield yield_sse_message("status", f"Paraphrase {k+1} generated.")
            except Exception as e:
                error_msg = f"Error generating Paraphrase {k+1}: {str(e)}"
                yield yield_sse_message("error", error_msg)
                paraphrased_prompts_list.append(f"Error generating paraphrase: {str(e)}")
        
        # --- First OpenAI Loop (Standard) ---
        openai_system_prompt_standard = "Return only your final answer, do NOT return any text showing thinking or chain of thought."
        yield yield_sse_message("status", "\n--- Starting Standard OpenAI Iterations ---")
        for i in range(10):
            current_overall_iteration = i + 1
            current_prompt_for_openai = original_prompt
            prompt_desc = "Original"
            if current_overall_iteration > 1 and (current_overall_iteration - 2) < len(paraphrased_prompts_list):
                current_prompt_for_openai = f"Original Prompt: {original_prompt}\nParaphrased Prompt: {paraphrased_prompts_list[current_overall_iteration - 2]}"
                prompt_desc = f"Original + Paraphrase #{current_overall_iteration - 1}"
            yield yield_sse_message("status", f"Running Standard OpenAI iteration {i+1}/10 ({prompt_desc})...")
            try:
                response = openai_client.chat.completions.create(model=openai_model_name, messages=[{"role": "system", "content": openai_system_prompt_standard}, {"role": "user", "content": current_prompt_for_openai}], temperature=(i * 0.1), max_tokens=1000)
                openai_results_dict[current_overall_iteration] = response.choices[0].message.content.strip()
                if response.usage:
                    total_openai_prompt_tokens += response.usage.prompt_tokens
                    total_openai_completion_tokens += response.usage.completion_tokens
                yield yield_sse_message("status", f"Standard OpenAI Iteration {i+1} completed.")
            except Exception as e:
                openai_results_dict[current_overall_iteration] = f"Error: {str(e)}"
                yield yield_sse_message("error", f"Error in Standard OpenAI API call (Iteration {i+1}): {str(e)}")

        # --- Second OpenAI Loop (Creative) ---
        openai_system_prompt_creative = "Return only your final answer, do NOT return any text showing thinking or chain of thought. Please be very creative."
        yield yield_sse_message("status", "\n--- Starting Creative OpenAI Iterations ---")
        for j in range(6):
            current_overall_iteration = 10 + j + 1
            current_prompt_for_openai = original_prompt
            prompt_desc = "Original (Fallback)"
            if (current_overall_iteration - 2) < len(paraphrased_prompts_list):
                current_prompt_for_openai = f"Original Prompt: {original_prompt}\nParaphrased Prompt: {paraphrased_prompts_list[current_overall_iteration - 2]}"
                prompt_desc = f"Original + Paraphrase #{current_overall_iteration - 1}"
            yield yield_sse_message("status", f"Running Creative OpenAI iteration {j+1}/6 ({prompt_desc})...")
            try:
                response = openai_client.chat.completions.create(model=openai_model_name, messages=[{"role": "system", "content": openai_system_prompt_creative}, {"role": "user", "content": current_prompt_for_openai}], temperature=((j * 0.1) + 0.5), max_tokens=1000)
                openai_results_dict[current_overall_iteration] = response.choices[0].message.content.strip()
                if response.usage:
                    total_openai_prompt_tokens += response.usage.prompt_tokens
                    total_openai_completion_tokens += response.usage.completion_tokens
                yield yield_sse_message("status", f"Creative OpenAI Iteration {j+1} completed.")
            except Exception as e:
                openai_results_dict[current_overall_iteration] = f"Error: {str(e)}"
                yield yield_sse_message("error", f"Error in Creative OpenAI API call (Iteration {j+1}): {str(e)}")

        # --- Gemini Consolidation (Streaming) ---
        yield yield_sse_message("status", "\nConsolidating all results with Gemini (streaming)...")
        gemini_system_prompt = "Please evaluate the user prompt and the results from calls to other LLMs. Consolidate all of the results into one answer, using the best ideas from each result. If you have better ideas that the ones provided, please add or change the consolidated result as you see fit. Only one final result should be returned."
        gemini_prompt_parts = [f"User Prompt: {original_prompt}\n\n--- OpenAI {openai_model_name} Iteration Results ---"]
        for k in sorted(openai_results_dict.keys()):
            gemini_prompt_parts.append(f"Iteration {k}:\n{openai_results_dict[k]}\n")
        
        full_gemini_prompt_text = "\n".join(gemini_prompt_parts)
        gemini_final_result_text_only = ""
        
        try:
            # Signal UI to create the container for the Gemini response
            yield yield_sse_message("final_result_start", "--- Gemini Final Answer ---", {"model_name": gemini_model_name})
            
            # Use stream=True to make the call non-blocking
            gemini_response_stream = gemini_model.generate_content(
                [gemini_system_prompt, full_gemini_prompt_text],
                stream=True
            )
            
            for chunk in gemini_response_stream:
                if chunk.text:
                    gemini_final_result_text_only += chunk.text
                    # Stream each chunk to the UI
                    yield yield_sse_message("final_result_chunk", chunk.text)
            
            # After streaming, get token counts from the resolved response object
            if hasattr(gemini_response_stream, 'usage_metadata') and gemini_response_stream.usage_metadata:
                 total_gemini_prompt_tokens = gemini_response_stream.usage_metadata.prompt_token_count
                 total_gemini_completion_tokens = gemini_response_stream.usage_metadata.candidates_token_count

        except Exception as e:
            error_msg = f"\nError in Gemini API call: {str(e)}"
            yield yield_sse_message("error", error_msg)
            gemini_final_result_text_only = error_msg 

        # --- Token Counts and Database Logging ---
        token_data_for_ui = {"openai": {"model": f"{openai_paraphrase_model_name} & {openai_model_name}", "input": total_openai_prompt_tokens, "output": total_openai_completion_tokens}, "gemini": {"model": gemini_model_name, "input": total_gemini_prompt_tokens, "output": total_gemini_completion_tokens}}
        yield yield_sse_message("token_summary", "Token Usage Summary", {"data": token_data_for_ui})
        
        token_summary_for_db = f"\n\n--- Token Usage ---\nOpenAI: Input: {total_openai_prompt_tokens}, Output: {total_openai_completion_tokens}\nGemini: Input: {total_gemini_prompt_tokens}, Output: {total_gemini_completion_tokens}"
        gemini_final_result_with_tokens_for_db = gemini_final_result_text_only + token_summary_for_db

        try:
            openai_results_json = json.dumps(openai_results_dict, indent=2) 
            save_log_to_db(original_prompt, openai_results_json, gemini_final_result_with_tokens_for_db) 
            yield yield_sse_message("status", "\nResults saved to database.")
        except Exception as e:
            yield yield_sse_message("error", f"Error saving to database: {str(e)}")
        
        yield f"event: stream_end\ndata: Processing complete.\n\n"

    return Response(stream_with_context(generate_responses_stream(user_prompt)), mimetype='text/event-stream')

# --- Main Execution ---
if __name__ == '__main__':
    init_db() 
    app.run(debug=True, threaded=True)

