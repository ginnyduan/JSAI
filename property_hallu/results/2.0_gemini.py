# %%
from dotenv import load_dotenv
import os, pathlib

ENV_PATH = str(pathlib.Path.home() / "llamp" / ".env.local")
load_dotenv(ENV_PATH, override=True)

for k in ["PMG_MAPI_KEY","MP_API_KEY","GOOGLE_API_KEY"]:
    v = os.getenv(k)
    print(k, "SET" if v else "MISSING", f"(len={len(v)})" if v else "")

# %%
import os, json, time, re, random
from pathlib import Path
import numpy as np
import pandas as pd

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
random.seed(42)
np.random.seed(42)

ROOT = Path(".")
DATA_DIR = ROOT / "data"
PROMPT_DIR = ROOT / "prompts"
LOG_DIR = ROOT / "logs"
RES_DIR = ROOT / "results"
ANN_DIR = ROOT / "annotation"

for d in [DATA_DIR, PROMPT_DIR, LOG_DIR, RES_DIR, ANN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print("RUN_ID =", RUN_ID)

# %%
from mp_api.client import MPRester

# 你之前用过：PMG_MAPI_KEY / MP_API_KEY
MP_KEY = os.getenv("PMG_MAPI_KEY") or os.getenv("MP_API_KEY")
assert MP_KEY, "Missing MP API key env: set PMG_MAPI_KEY or MP_API_KEY"

from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_MODEL = "models/gemini-2.5-flash-lite"  # 你 list_models 里有这个
top_llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=0.0,
    convert_system_message_to_human=True,  # 避免 SystemMessage 报错
)
print("top_llm built:", GEMINI_MODEL)

# %%
# Cell 1: Setup & Configuration
import os
import pathlib
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# 1. Load environment variables (keep path consistent with your part1.py)
ENV_PATH = str(pathlib.Path.home() / "llamp" / ".env.local")
load_dotenv(ENV_PATH, override=True)

# 2. Get API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY is missing, please check your .env file")

# 3. Configure native SDK
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 🎯 Core Correction: Use your confirmed Gemini 2.5
# ==========================================
MODEL_NAME = "models/gemini-2.5-flash-lite" 

print(f"Initializing model: {MODEL_NAME} (Temperature=0.0) ...")

try:
    # Initialize model
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.0} # Keep T=0.0 to match your previous experiments
    )
    print(f"✅ Model loaded successfully! Using: {MODEL_NAME}")
    
except Exception as e:
    print(f"\n❌ Model initialization error: {e}")
    print("Please check if your API Key has access to this model version.")

# Prepare result saving directory
RES_DIR = Path("results_cove")
RES_DIR.mkdir(exist_ok=True)
print(f"📂 Results will be saved to: {RES_DIR.resolve()}")

# %%
import pandas as pd
import os

# === Configure input file path ===
# Based on part1.py output, filename might be "gt_200.csv" or "results/gt_200.csv"
# Modify here if not found
INPUT_FILE = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv")

df_gt = pd.read_csv(INPUT_FILE)
# === Data Preprocessing: Convert to test task list ===
# part1.py data is "wide" (one row contains material_id, band_gap, formation_energy_per_atom)
# We need to convert it to "long" format (test one property at a time) for CoV loop processing

tasks = []
# ⚠️ Debug Mode: Only take the first 5 materials for testing.
# If you want to run the full 200, delete .head(5)!
target_samples = df_gt.head(5) 

for _, row in target_samples.iterrows():
    mp_id = row['material_id']
    
    # Task 1: Band Gap
    if pd.notna(row.get('band_gap')):
        tasks.append({
            "material_id": mp_id,
            "property": "band_gap",
            "ground_truth": row['band_gap']
        })
        
    # Task 2: Formation Energy
    # Note: column name might be 'formation_energy_per_atom' or 'gt_fe_per_atom', adjust based on part1.py
    fe_col = 'formation_energy_per_atom' if 'formation_energy_per_atom' in row else 'gt_fe_per_atom'
    if pd.notna(row.get(fe_col)):
        tasks.append({
            "material_id": mp_id,
            "property": "formation_energy_per_atom",
            "ground_truth": row[fe_col]
        })

df_tasks = pd.DataFrame(tasks)
print(f"✅ Data loaded! Ready to run {len(df_tasks)} CoV experiments.")
print("Preview of first 3 tasks:")
display(df_tasks.head(3))

# %%
def retrieve_with_verification(mp_id, property_name):
    """
    Chain of Verification (CoV) Two-Step Inference Function
    """
    
    # --- Step 1: Initial Query (Generator) ---
    # Keep the prompt simple similar to part1.py to induce errors
    if property_name == "band_gap":
        unit = "eV"
    else:
        unit = "eV/atom"
        
    initial_prompt = f"What is the {property_name} ({unit}) of material {mp_id}? Answer with a single number."
    
    try:
        # Use the model object initialized in Cell 1
        # T=0.0 ensures reproducibility
        response1 = model.generate_content(
            initial_prompt, 
            generation_config={"temperature": 0.0}
        )
        initial_output = response1.text.strip()
    except Exception as e:
        return f"Gen_Error: {e}", "Error"

    # --- Step 2: Self-Verification (Verifier) ---
    # CoV Core: Use Step 1 output as context and ask model to "clean" the data
    verification_prompt = f"""
    Role: You are a rigorous scientific data parser.
    
    Task: Examine the 'Model Response' below and extract the extracted numerical value for {property_name}.
    
    Input Context:
    - User Query: "{initial_prompt}"
    - Model Response: "{initial_output}"
    
    Verification Rules:
    1. Extract ONLY the numeric value corresponding to the property.
    2. CRITICAL: Do NOT extract digits from the Material ID (e.g., if ID is 'mp-619575', do not output '619575').
    3. If the response contains a unit (e.g., '1.5 eV'), remove the unit.
    4. If the response is a refusal (e.g., 'I don't know'), output 'NaN'.
    
    Final Output: Return ONLY the number (float).
    """
    
    try:
        response2 = model.generate_content(
            verification_prompt, 
            generation_config={"temperature": 0.0}
        )
        verified_output = response2.text.strip()
    except Exception as e:
        verified_output = f"Ver_Error: {e}"

    return initial_output, verified_output

print("✅ CoV Function Defined!")

# %%
import time

results = []

print(f"🚀 Starting CoV Experiment (Total {len(df_tasks)} tasks)...")
print("-" * 60)

for index, row in df_tasks.iterrows():
    mp_id = row['material_id']
    prop = row['property']
    gt = row['ground_truth']
    
    # Execute CoV
    raw_out, verified_out = retrieve_with_verification(mp_id, prop)
    
    # Simple "is rescued" flag (Step 1 != Step 2)
    # Note: This is just a string comparison, you may need numeric comparison scripts later
    is_changed = (raw_out.strip() != verified_out.strip())
    
    results.append({
        "material_id": mp_id,
        "property": prop,
        "ground_truth": gt,
        "step1_raw": raw_out,
        "step2_verified": verified_out,
        "is_changed": is_changed
    })
    
    # Print progress (print once per completion for easy monitoring)
    print(f"[{index+1}/{len(df_tasks)}] {mp_id} ({prop})")
    if is_changed:
        print(f"    ✨ CHANGE DETECTED: '{raw_out}' -> '{verified_out}'")
    else:
        print(f"    (No change: '{verified_out}')")
        
    # Pause for 1 second to avoid rate limits
    time.sleep(1.0)

print("-" * 60)
print("✅ Experiment Finished!")

# %%
# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to results_cove directory
output_file = RES_DIR / "cove_experiment_results.csv"
df_results.to_csv(output_file, index=False)
print(f"💾 Results saved to: {output_file}")

# Show cases that were "modified" (Evidence of Entanglement Rescue)
print("\n=== CoV Rescue Cases (Step 1 != Step 2) ===")
changed_cases = df_results[df_results['is_changed'] == True]

if len(changed_cases) > 0:
    display(changed_cases[['material_id', 'property', 'step1_raw', 'step2_verified']])
else:
    print("No modified outputs found (Model might be perfect in Step 1, or Verifier didn't work).")

# Show overall preview (first 5 rows)
print("\n=== Overall Data Preview ===")
display(df_results.head())

# %% [markdown]
# Full Run (400 in total)

# %%
import pandas as pd
import os

# === 1. Configure Input File Path ===
# Attempt to find the file generated by part1.py
INPUT_FILE = "gt_200.csv" 

if not os.path.exists(INPUT_FILE):
    # If not in current directory, try looking in 'results' directory
    if os.path.exists(f"results/{INPUT_FILE}"):
        INPUT_FILE = f"results/{INPUT_FILE}"
    else:
        # If still not found, raise error
        raise FileNotFoundError(f"❌ Cannot find {INPUT_FILE}! Please confirm if part1.py ran successfully and generated this file.")

print(f"📖 Reading data: {INPUT_FILE} ...")

# === 2. Key Step: Define df_gt ===
# (Reads the CSV into a DataFrame)
df_gt = pd.read_csv(INPUT_FILE) 

# === 3. Data Preprocessing: Convert to Test Task List ===
tasks = []

# ==========================================
# 🚀 FULL RUN CONFIGURATION
# Using the full dataset (no .head() limit)
# ==========================================
target_samples = df_gt  

print(f"Preparing to process {len(target_samples)} materials...")

for _, row in target_samples.iterrows():
    mp_id = row['material_id']
    
    # Task 1: Band Gap
    if pd.notna(row.get('band_gap')):
        tasks.append({
            "material_id": mp_id,
            "property": "band_gap",
            "ground_truth": row['band_gap']
        })
        
    # Task 2: Formation Energy
    # Compatible column names: might be 'formation_energy_per_atom' or 'gt_fe_per_atom'
    fe_col = 'formation_energy_per_atom' if 'formation_energy_per_atom' in row else 'gt_fe_per_atom'
    if pd.notna(row.get(fe_col)):
        tasks.append({
            "material_id": mp_id,
            "property": "formation_energy_per_atom",
            "ground_truth": row[fe_col]
        })

df_tasks = pd.DataFrame(tasks)
print(f"✅ Data loaded! Ready to run {len(df_tasks)} CoV experiments.")
# Should display around 400 tasks

# %%
def retrieve_with_verification(mp_id, property_name):
    """
    Chain of Verification (CoV) Two-Step Inference Function
    """
    
    # --- Step 1: Initial Query (Generator) ---
    # Keep the prompt simple similar to part1.py to induce errors
    if property_name == "band_gap":
        unit = "eV"
    else:
        unit = "eV/atom"
        
    initial_prompt = f"What is the {property_name} ({unit}) of material {mp_id}? Answer with a single number."
    
    try:
        # Use the model object initialized in Cell 1
        # T=0.0 ensures reproducibility
        response1 = model.generate_content(
            initial_prompt, 
            generation_config={"temperature": 0.0}
        )
        initial_output = response1.text.strip()
    except Exception as e:
        return f"Gen_Error: {e}", "Error"

    # --- Step 2: Self-Verification (Verifier) ---
    # CoV Core: Use Step 1 output as context and ask model to "clean" the data
    verification_prompt = f"""
    Role: You are a rigorous scientific data parser.
    
    Task: Examine the 'Model Response' below and extract the extracted numerical value for {property_name}.
    
    Input Context:
    - User Query: "{initial_prompt}"
    - Model Response: "{initial_output}"
    
    Verification Rules:
    1. Extract ONLY the numeric value corresponding to the property.
    2. CRITICAL: Do NOT extract digits from the Material ID (e.g., if ID is 'mp-619575', do not output '619575').
    3. If the response contains a unit (e.g., '1.5 eV'), remove the unit.
    4. If the response is a refusal (e.g., 'I don't know'), output 'NaN'.
    
    Final Output: Return ONLY the number (float).
    """
    
    try:
        response2 = model.generate_content(
            verification_prompt, 
            generation_config={"temperature": 0.0}
        )
        verified_output = response2.text.strip()
    except Exception as e:
        verified_output = f"Ver_Error: {e}"

    return initial_output, verified_output

print("✅ CoV Function Defined!")

# %%
import time

results = []

print(f"🚀 Starting CoV Experiment (Total {len(df_tasks)} tasks)...")
print("-" * 60)

for index, row in df_tasks.iterrows():
    mp_id = row['material_id']
    prop = row['property']
    gt = row['ground_truth']
    
    # Execute CoV
    raw_out, verified_out = retrieve_with_verification(mp_id, prop)
    
    # Simple "is rescued" flag (Step 1 != Step 2)
    # Note: This is just a string comparison; you may need numeric comparison scripts later
    is_changed = (raw_out.strip() != verified_out.strip())
    
    results.append({
        "material_id": mp_id,
        "property": prop,
        "ground_truth": gt,
        "step1_raw": raw_out,
        "step2_verified": verified_out,
        "is_changed": is_changed
    })
    
    # Print progress (print once per completion for easy monitoring)
    print(f"[{index+1}/{len(df_tasks)}] {mp_id} ({prop})")
    if is_changed:
        print(f"    ✨ CHANGE DETECTED: '{raw_out}' -> '{verified_out}'")
    else:
        print(f"    (No change: '{verified_out}')")
        
    # Pause for 1 second to avoid rate limits
    time.sleep(1.0)

print("-" * 60)
print("✅ Experiment Finished!")

# %%
# Convert to DataFrame
df_results = pd.DataFrame(results)

# Save to results_cove directory
output_file = RES_DIR / "cove_experiment_results.csv"
df_results.to_csv(output_file, index=False)
print(f"💾 Results saved to: {output_file}")

# Show cases that were "modified" (Evidence of Entanglement Rescue)
print("\n=== CoV Rescue Cases (Step 1 != Step 2) ===")
changed_cases = df_results[df_results['is_changed'] == True]

if len(changed_cases) > 0:
    display(changed_cases[['material_id', 'property', 'step1_raw', 'step2_verified']])
else:
    print("No modified outputs found (Model might be perfect in Step 1, or Verifier didn't work).")

# Show overall preview (first 5 rows)
print("\n=== Overall Data Preview ===")
display(df_results.head())

# %% [markdown]
# Tool + Cov
# 

# %%
import time
import pandas as pd

# === Define Tool + CoV Function ===
def retrieve_with_tool_and_cove(mp_id, property_name, tool_value):
    """
    Methodology 2: 
    1. Provide the model with the correct Tool Value.
    2. Ask it to generate a natural language response (inducing potential ID entanglement).
    3. Use CoV to "clean" the response.
    """
    
    # --- Round 1: Simulated RAG Generation ---
    # (Deliberately vague instructions to check for entanglement)
    prompt_r1 = f"""
    Context: The user asks for {property_name} of {mp_id}.
    Tool Result: The database returned {tool_value}.
    
    Task: Answer the user's question using the tool result. 
    (Please answer in a full sentence).
    """
    
    try:
        resp1 = model.generate_content(prompt_r1, generation_config={"temperature": 0.0})
        text_r1 = resp1.text.strip()
    except Exception as e:
        return f"Error: {e}", "Error"

    # --- Round 2: CoV Verification (The Soft Fix) ---
    # (Specifically targets cleaning up ID entanglement)
    prompt_r2 = f"""
    Task: Extract the clean numerical value from the Model Response.
    
    Model Response: "{text_r1}"
    
    Rules:
    1. Extract ONLY the value matching the Tool Result.
    2. CRITICAL: Do NOT extract the digits from Material ID ({mp_id}).
    3. Remove units. Return ONLY the number.
    """
    
    try:
        resp2 = model.generate_content(prompt_r2, generation_config={"temperature": 0.0})
        text_r2 = resp2.text.strip()
    except Exception as e:
        text_r2 = f"Error: {e}"
        
    return text_r1, text_r2

# === Run Experiment ===
results_tool_cove = []
print(f"🚀 Starting Methodology 2: Tool + CoV (Soft Fix) Experiment...")
print("-" * 60)

for index, row in df_tasks.iterrows():
    mp_id = row['material_id']
    prop = row['property']
    gt = row['ground_truth'] # Simulated Tool Value
    
    # Execute
    r1_out, r2_out = retrieve_with_tool_and_cove(mp_id, prop, gt)
    
    # Record results
    results_tool_cove.append({
        "mp_id": mp_id,
        "tool_value": gt,
        "round1_natural_language": r1_out,
        "round2_cove_extracted": r2_out
    })
    
    # Print progress (every 20 items)
    if (index + 1) % 20 == 0:
        print(f"[{index + 1}/{len(df_tasks)}] Processed...")
        # Check for potential entanglement fixes during runtime
        # (Heuristic: ID digits present in Round 1 but absent in Round 2)
        id_num = mp_id.replace("mp-", "")
        if id_num in r1_out and id_num not in r2_out:
            print(f"    🌟 Entanglement Fix Detected! R1: {r1_out} -> R2: {r2_out}")
            
    time.sleep(0.5)

# === Save Results ===
df_tool_cove = pd.DataFrame(results_tool_cove)
output_path = RES_DIR / "tool_cove_results.csv"
df_tool_cove.to_csv(output_path, index=False)
print("-" * 60)
print(f"✅ Methodology 2 Finished! Results saved to {output_path}")

# %% [markdown]
# Structured_outputs_method (avoid parser)

# %%
# === 1. Re-import Libraries & Setup ===
import os
import pathlib
import pandas as pd
import time
import typing_extensions as typing
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load Env
ENV_PATH = str(pathlib.Path.home() / "llamp" / ".env.local")
load_dotenv(ENV_PATH, override=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Model (Gemini 2.5)
MODEL_NAME = "models/gemini-2.5-flash-lite"
try:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        generation_config={"temperature": 0.0}
    )
    print(f"✅ Model restored: {MODEL_NAME}")
except Exception as e:
    print(f"❌ Model Error: {e}")

# Result Directory
RES_DIR = Path("results_cove")
RES_DIR.mkdir(exist_ok=True)

# === 2. Restore Data (The part you missed) ===
INPUT_FILE = Path("/Users/yutongduan/llamp/experiments/property_hallu/results/gt_200_full.csv")

df_gt = pd.read_csv(INPUT_FILE)

# Re-create the tasks list (Logic from Cell 2)
tasks = []
target_samples = df_gt # Full run

for _, row in target_samples.iterrows():
    mp_id = row['material_id']
    # Task 1: Band Gap
    if pd.notna(row.get('band_gap')):
        tasks.append({"material_id": mp_id, "property": "band_gap", "ground_truth": row['band_gap']})
    # Task 2: Formation Energy
    fe_col = 'formation_energy_per_atom' if 'formation_energy_per_atom' in row else 'gt_fe_per_atom'
    if pd.notna(row.get(fe_col)):
        tasks.append({"material_id": mp_id, "property": "formation_energy_per_atom", "ground_truth": row[fe_col]})

df_tasks = pd.DataFrame(tasks)
print(f"✅ Data restored! `df_tasks` has {len(df_tasks)} rows.")
print("👉 You can now run the Structured Output experiment immediately.")

# %%
# Cell 7: Methodology 3 - Structured Output Experiment (Hard Fix)
# (Uses JSON Schema to enforce strictly formatted outputs, eliminating entanglement)

import typing_extensions as typing
import json
import time
import pandas as pd

# === 1. Define the Schema (The "Answer Sheet") ===
# We define a strict shape: a JSON object with a single float field 'extracted_value'.
# This acts as a "Hard Constraint" during the model's token generation.
class MaterialProperty(typing.TypedDict):
    extracted_value: float

# === 2. Define the Structured Output Function ===
def retrieve_with_structured_output(mp_id, property_name, tool_value):
    """
    Methodology 3: 
    1. Input: The correct value from the Tool (simulated by passing GT).
    2. Constraint: Force JSON output using 'response_schema'.
    3. Goal: Verify if this eliminates ID-Value entanglement completely.
    """
    
    # Prompt: We give the model the correct answer (from tool) 
    # and ask it to format it.
    prompt = f"""
    You are a data extraction interface connecting to the Materials Project API.
    
    Context:
    - User asked for: {property_name} of {mp_id}
    - The API Tool returned the value: {tool_value}
    
    Task:
    Output the value strictly in JSON format.
    """
    
    try:
        # 🌟 CORE MECHANISM: response_schema enforcement 🌟
        # This tells the model: "Do not output text. Output JSON matching MaterialProperty."
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json", 
                "response_schema": MaterialProperty 
            }
        )
        return response.text
    except Exception as e:
        return f'{{"error": "{e}"}}'

# === 3. Run the Experiment ===
structured_results = []
print(f"🚀 Starting Methodology 3: Structured Output Experiment...")
print("-" * 60)

for index, row in df_tasks.iterrows():
    mp_id = row['material_id']
    prop = row['property']
    # We use the ground_truth from CSV as the "Tool Output"
    # This simulates a perfect RAG retrieval from the MP API.
    tool_value_input = row['ground_truth'] 
    
    # Execute
    json_output = retrieve_with_structured_output(mp_id, prop, tool_value_input)
    
    # Parse logic
    extracted_val = None
    is_perfect = False
    
    try:
        # Parse the JSON string
        parsed = json.loads(json_output)
        
        # Handle cases where model might output list or dict
        if isinstance(parsed, list):
            parsed = parsed[0]
            
        extracted_val = parsed.get("extracted_value")
        
        # Verification: Did the model echo the number correctly?
        if extracted_val is not None:
            # Allow tiny float tolerance (1e-6) for floating point comparison
            # In a "copy-paste" task, it should be exact.
            is_perfect = abs(float(extracted_val) - float(tool_value_input)) < 1e-4
            
    except Exception as e:
        extracted_val = f"JSON_Parse_Error: {e}"
        is_perfect = False
    
    structured_results.append({
        "material_id": mp_id,
        "property": prop,
        "tool_input_value": tool_value_input,
        "model_json_output": json_output,
        "final_extracted": extracted_val,
        "is_perfect_match": is_perfect
    })
    
    # Print progress every 20 items
    if (index + 1) % 20 == 0:
        print(f"[{index + 1}/{len(df_tasks)}] Processed... Last Result: Match={is_perfect}")
    
    # Fast sleep (0.2s) as this is usually faster and less prone to hallucinations
    time.sleep(0.2)

print("-" * 60)
print("✅ Methodology 3 Finished!")

# === 4. Save and Analyze ===
df_struct = pd.DataFrame(structured_results)
output_path = RES_DIR / "structured_output_results.csv"
df_struct.to_csv(output_path, index=False)

# Calculate Success Rate
success_rate = df_struct['is_perfect_match'].mean()
print(f"\n🏆 Final Perfect Match Rate: {success_rate:.2%}")

if success_rate > 0.99:
    print("Conclusion: Structured Outputs successfully eliminated entanglement!")
else:
    print("Warning: Some errors persist. Please inspect the CSV.")

# %%
# 强制更新到最新版
!pip install -U google-generativeai pydantic

# %%



