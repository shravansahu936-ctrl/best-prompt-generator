import streamlit as st
import google.generativeai as genai
from st_copy_button import st_copy_button
from streamlit_mic_recorder import speech_to_text
import sqlite3
import pandas as pd
import json

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Best Prompt Generator", page_icon="🚀", layout="wide")

# --- Helper: Dynamic Working Model Selector ---
def get_active_model():
    """Automatically selects the recommended gemini-3.8-flash model."""
    target_fallback = "gemini-3.8-flash"
    try:
        available_models = [
            m.name for m in genai.list_models() 
            if "generateContent" in m.supported_generation_methods
        ]
        
        candidates = [
            "gemini-3.8-flash",
            "models/gemini-3.8-flash",
            "gemini-2.0-flash",
            "gemini-1.5-flash-latest"
        ]
        
        for cand in candidates:
            for m_name in available_models:
                if "2.5" in m_name:
                    continue
                if cand in m_name or m_name.endswith(cand):
                    return genai.GenerativeModel(m_name)
                    
        valid_models = [m for m in available_models if "2.5" not in m]
        if valid_models:
            return genai.GenerativeModel(valid_models[0])
    except Exception:
        pass
        
    return genai.GenerativeModel(target_fallback)

# --- Database Setup (SQLite Integration) ---
@st.cache_resource
def init_db():
    conn = sqlite3.connect("prompts_database.db", check_same_thread=False)
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT,
                prompt TEXT,
                category TEXT,
                target_model TEXT,
                tone TEXT,
                output_format TEXT,
                output_language TEXT,
                creativity REAL,
                favorite INTEGER,
                rating TEXT
            )
        ''')
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(prompts)")
        columns = [info[1] for info in cursor.fetchall()]
        if "rating" not in columns:
            conn.execute("ALTER TABLE prompts ADD COLUMN rating TEXT")
    return conn

conn = init_db()

# --- Session States ---
if "user_text" not in st.session_state:
    st.session_state.user_text = ""
if "prompts_variations" not in st.session_state:
    st.session_state.prompts_variations = {}
if "selected_variation_key" not in st.session_state:
    st.session_state.selected_variation_key = "Option A"
if "current_prompt_id" not in st.session_state:
    st.session_state.current_prompt_id = None
if "test_output" not in st.session_state:
    st.session_state.test_output = ""

def set_template(text):
    st.session_state.user_text = text

# --- Secure API Configuration (No Key Exposure) ---
api_key_configured = False

if "GEMINI_API_KEY" in st.secrets and st.secrets["GEMINI_API_KEY"]:
    active_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=active_key)
    api_key_configured = True
else:
    with st.sidebar:
        st.markdown("### 🔑 API Configuration")
        user_entered_key = st.text_input(
            "Gemini API Key",
            value="",
            type="password",
            help="Enter your Gemini API key here."
        )
        if user_entered_key:
            genai.configure(api_key=user_entered_key)
            api_key_configured = True
        else:
            st.warning("Please configure your Gemini API Key in Secrets or enter it above.")

# --- Custom Styling ---
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1614729939124-032f0b56c9ce?q=80&w=2000&auto=format&fit=crop"); 
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    [data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.85) !important;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 255, 255, 0.2);
    }
    [data-testid="stHeader"] {
        background-color: rgba(0,0,0,0);
    }
    .main-header {
        font-size: 3rem !important;
        font-weight: 900 !important;
        color: #ffffff; 
        text-shadow: 0 0 15px #0ff, 2px 2px 5px rgba(0,0,0,0.9);
        margin-bottom: 0px;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem !important;
        color: #e0f2fe;
        text-shadow: 1px 1px 5px rgba(0,0,0,0.9);
        margin-bottom: 25px;
        text-align: center;
    }
    div[data-testid="stTextArea"] textarea {
        border-radius: 12px;
        font-size: 1.1rem;
        background-color: rgba(15, 23, 42, 0.65) !important;
        backdrop-filter: blur(10px);
        color: #f8fafc;
        border: 1px solid rgba(0, 255, 255, 0.3);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.6);
    }
    .stButton>button {
        border-radius: 10px;
        height: 52px;
        font-size: 1.1rem;
        font-weight: bold;
        background: linear-gradient(90deg, #0284c7, #8b5cf6, #0ea5e9);
        border: none;
        color: white;
        transition: 0.5s;
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.6);
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(0, 255, 255, 0.8);
    }
    h3, p, .stMarkdown, .stSelectbox label, .stSelectbox div, .stSlider div {
        color: #ffffff !important;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.9);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Database Helpers ---
def get_prompts_from_db():
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, query, prompt, category, target_model, tone, output_format, 
               output_language, creativity, favorite, rating 
        FROM prompts ORDER BY id DESC
    """)
    rows = cursor.fetchall()
    return [
        {
            "id": r[0], "query": r[1], "prompt": r[2], "category": r[3],
            "target_model": r[4], "tone": r[5], "output_format": r[6],
            "output_language": r[7], "creativity": r[8], "favorite": bool(r[9]), "rating": r[10]
        }
        for r in rows
    ]

def add_prompt_to_db(item):
    with conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO prompts (query, prompt, category, target_model, tone, output_format, output_language, creativity, favorite, rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (item['query'], item['prompt'], item['category'], item['target_model'], item['tone'], item['output_format'], item['output_language'], item['creativity'], int(item['favorite']), item['rating']))
        return cursor.lastrowid

def update_favorite_in_db(prompt_id, status):
    with conn:
        conn.execute("UPDATE prompts SET favorite = ? WHERE id = ?", (int(status), prompt_id))

def update_prompt_text_in_db(prompt_id, new_prompt):
    with conn:
        conn.execute("UPDATE prompts SET prompt = ? WHERE id = ?", (new_prompt, prompt_id))

def clear_db():
    with conn:
        conn.execute("DELETE FROM prompts")

# --- Refine Prompt Function ---
def refine_prompt_text(instruction_type, target_key):
    if not api_key_configured:
        st.error("API Key configure nahi hai.")
        return
    current_text = st.session_state.prompts_variations.get(target_key, "")
    if current_text:
        with st.spinner(f"Refining ({target_key} - {instruction_type})..."):
            modifier = {
                "shorter": "Make this prompt much more concise while retaining critical constraints.",
                "detailed": "Expand this prompt by adding deeper domain details and structural expectations.",
                "professional": "Elevate this prompt tone to be exceptionally professional and authoritative."
            }.get(instruction_type, "")
            
            refinement_query = f"{modifier}\n\nOriginal Prompt:\n{current_text}"
            
            try:
                model = get_active_model()
                response = model.generate_content(refinement_query)
                refined = response.text.strip()
                st.session_state.prompts_variations[target_key] = refined
                st.session_state.test_output = ""
                if st.session_state.current_prompt_id:
                    update_prompt_text_in_db(st.session_state.current_prompt_id, refined)
            except Exception as e:
                st.error(f"Refinement error: {e}")

# --- Test Run Generated Prompt ---
def test_run_prompt(prompt_to_run):
    if not api_key_configured:
        st.error("API Key configure nahi hai.")
        return
    if prompt_to_run:
        with st.spinner("⚡ Running prompt through Gemini to get AI answer..."):
            try:
                model = get_active_model()
                response = model.generate_content(prompt_to_run)
                st.session_state.test_output = response.text.strip()
            except Exception as e:
                st.error(f"Execution Error: {e}")

# --- Helper: Split Multi-Variation Output ---
def parse_variations(raw_text):
    variations = {
        "Option A (Concise & Direct)": "",
        "Option B (Persona & Framework)": "",
        "Option C (Few-Shot / Examples)": ""
    }
    parts_a = raw_text.split("===OPTION_A===")
    if len(parts_a) > 1:
        parts_b = parts_a[1].split("===OPTION_B===")
        variations["Option A (Concise & Direct)"] = parts_b[0].strip()
        if len(parts_b) > 1:
            parts_c = parts_b[1].split("===OPTION_C===")
            variations["Option B (Persona & Framework)"] = parts_c[0].strip()
            if len(parts_c) > 1:
                variations["Option C (Few-Shot / Examples)"] = parts_c[1].strip()
    else:
        variations["Option A (Concise & Direct)"] = raw_text.strip()
        variations["Option B (Persona & Framework)"] = raw_text.strip()
        variations["Option C (Few-Shot / Examples)"] = raw_text.strip()
    return variations

# --- Sidebar UI ---
with st.sidebar:
    st.markdown("### 📜 Prompts Manager")
    tab_hist, tab_fav, tab_export = st.tabs(["🕒 History", "⭐ Favorites", "📥 Export"])
    
    prompt_history = get_prompts_from_db()
    
    with tab_hist:
        if not prompt_history:
            st.info("Aapki history yahan show hogi.")
        else:
            for item in prompt_history:
                short_query = (item['query'][:22] + "...") if len(item['query']) > 22 else item['query']
                fav_icon = "⭐" if item['favorite'] else "☆"
                with st.expander(f"{fav_icon} {short_query}"):
                    st.caption(f"**Target:** {item['target_model']} | **Lang:** {item['output_language']}")
                    st.code(item['prompt'], language="markdown")
                    if st.button("⭐ Toggle Favorite", key=f"fav_hist_{item['id']}"):
                        update_favorite_in_db(item['id'], not item['favorite'])
                        st.rerun()
                        
    with tab_fav:
        favorites = [item for item in prompt_history if item['favorite']]
        if not favorites:
            st.info("No favorite prompts saved yet.")
        else:
            for item in favorites:
                short_query = (item['query'][:25] + "...") if len(item['query']) > 25 else item['query']
                with st.expander(f"⭐ {short_query}"):
                    st.caption(f"**Category:** {item['category']}")
                    st.code(item['prompt'], language="markdown")

    with tab_export:
        if prompt_history:
            json_data = json.dumps(prompt_history, indent=4)
            st.download_button(
                label="📥 Download as JSON",
                data=json_data,
                file_name="prompts_history.json",
                mime="application/json",
                use_container_width=True
            )
            df = pd.DataFrame(prompt_history)
            st.download_button(
                label="📥 Download as CSV",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name="prompts_history.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    st.divider()
    if st.button("🗑️ Clear Database History", use_container_width=True):
        clear_db()
        st.session_state.prompts_variations = {}
        st.session_state.current_prompt_id = None
        st.session_state.test_output = ""
        st.rerun()

# --- Main Layout ---
st.markdown('<p class="main-header">🚀 Best Prompt Generator</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Harness the power of AI. Enter your ideas to build master prompts.</p>', unsafe_allow_html=True)
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### ⚙️ Advanced Customization")
    mode = st.selectbox("Select Category:", [
        "🎓 Student / Academic", "💼 Professional / Business", "🎬 YouTube / Creative", 
        "🤖 Developer / Coding", "🎨 Image Generation", "🗣️ Everyday Advice"
    ])
    target_model = st.selectbox("🤖 Target AI Model:", [
        "ChatGPT (GPT-4o)", "Claude 3.5 Sonnet", "Gemini 1.5 / 2.0", "Midjourney / DALL-E 3"
    ])
    tone = st.selectbox("🎭 Tone & Persona:", [
        "Formal & Professional", "Beginner-Friendly & Simple", "Casual & Conversational", 
        "Direct & Concise", "Sarcastic & Witty", "Academic"
    ])
    output_format = st.selectbox("📊 Desired Output Format:", [
        "Step-by-Step Guide", "Bullet Points & Key Takeaways", "Markdown Table", 
        "Clean Code Only with Comments", "Short Summary", "Detailed Essay/Report"
    ])
    output_language = st.selectbox("🌐 Master Prompt Language:", [
        "English", "Hindi", "Hinglish", "Urdu", "Marathi", "Gujarati"
    ])
    
    st.markdown("---")
    st.markdown("#### 🎛️ Creativity & Tone")
    creativity_level = st.slider("Adjust Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

with col2:
    st.markdown("### 📝 Your Problem / Idea")
    
    # --- Voice Input & Text Area Integration ---
    voice_col, info_col = st.columns([1, 3])
    with voice_col:
        # Voice to text input (supports Hindi, English, Hinglish)
        audio_text = speech_to_text(
            language='en-IN',
            start_prompt="🎙️ Speak Idea",
            stop_prompt="⏹️ Stop Recording",
            key='voice_recorder'
        )
        if audio_text:
            st.session_state.user_text = audio_text
            st.rerun()
            
    with info_col:
        st.caption("🎙️ Mic click karke bolen ya neeche type karein.")

    user_problem = st.text_area(
        "Label hidden",
        value=st.session_state.user_text,
        placeholder="E.g., Mujhe kal ek presentation deni hai cost accounting pe, points samajh nahi aa rahe...",
        height=140,
        label_visibility="collapsed",
        key="user_text_input"
    )
    # Sync manual typing back to state
    st.session_state.user_text = user_problem
    
    st.caption("✨ Try a quick example:")
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        st.button("📝 Essay Help", on_click=set_template, args=("Mujhe 'Impact of Artificial Intelligence' par ek 500 words ka essay chahiye. Strong intro aur conclusion chahiye.",), use_container_width=True)
    with t_col2:
        st.button("💻 Python Error", on_click=set_template, args=("Mera Python script bar bar 'IndexError: list index out of range' error de raha hai. Isko fix kaise karein?",), use_container_width=True)
    with t_col3:
        st.button("🎬 YT Script", on_click=set_template, args=("Ek 5-minute ki YouTube video banani hai 'Top 5 Budget Smartphones in 2026' par. Catchy script structure chahiye.",), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ Generate 3 Master Prompt Variations", type="primary", use_container_width=True)

SYSTEM_PROMPT = f"""
You are an Expert AI Prompt Engineer. Transform the user's raw request into 3 distinct, high-quality prompt variations.

Target Configuration:
- Category: {mode}
- Target AI Model: {target_model}
- Desired Tone: {tone}
- Desired Output Format: {output_format}
- Language: {output_language}

CRITICAL RULES:
1. Do NOT solve the user's request directly. ONLY generate prompts ready for copy/pasting into an AI.
2. The entire output MUST be written in {output_language}.
3. You MUST format the response strictly with these exact separator delimiters:

===OPTION_A===
[Generate Option A: Concise & Direct - Short, compact, task-focused prompt with minimal fluff]

===OPTION_B===
[Generate Option B: Persona & Framework-Based - Uses Role, Context, Task, Tone, Guidelines, and Format structure]

===OPTION_C===
[Generate Option C: Few-Shot / Examples - Includes precise task context plus 1-2 realistic input/output examples demonstrating the desired response structure]
"""

if generate_btn:
    if not api_key_configured:
        st.error("⚠️ Gemini API key configure nahi hai.")
    elif user_problem.strip():
        with st.spinner(f"Crafting 3 master variations in {output_language}..."):
            try:
                full_prompt = f"{SYSTEM_PROMPT}\n\nUser Request:\n{user_problem}"
                
                model = get_active_model()
                response = model.generate_content(
                    full_prompt,
                    generation_config={"temperature": creativity_level}
                )
                
                if response.text:
                    variations = parse_variations(response.text)
                    st.session_state.prompts_variations = variations
                    st.session_state.test_output = ""
                    
                    primary_prompt = variations.get("Option B (Persona & Framework)") or list(variations.values())[0]
                    new_item = {
                        "query": user_problem,
                        "prompt": primary_prompt,
                        "category": mode,
                        "target_model": target_model,
                        "tone": tone,
                        "output_format": output_format,
                        "output_language": output_language,
                        "creativity": creativity_level,
                        "favorite": False,
                        "rating": None
                    }
                    new_id = add_prompt_to_db(new_item)
                    st.session_state.current_prompt_id = new_id
                    st.rerun()
                else:
                    st.error("Model did not return text. Please try again.")
            except Exception as e:
                st.error(f"API Error: {e}")
    else:
        st.warning("⚠️ Text box mein problem likhein ya voice se bole.")

# --- Multi-Variation Display Section ---
if st.session_state.prompts_variations:
    st.divider()
    st.markdown("### 🎯 Generated Master Prompt Variations (A/B Options)")
    
    tab_titles = list(st.session_state.prompts_variations.keys())
    tabs = st.tabs(tab_titles)
    
    for idx, tab in enumerate(tabs):
        opt_key = tab_titles[idx]
        current_text = st.session_state.prompts_variations[opt_key]
        
        with tab:
            words_count = len(current_text.split())
            tokens_count = int(len(current_text) / 3.8)
            
            m1, m2, _ = st.columns([1, 1, 2])
            m1.metric("📝 Word Count", words_count)
            m2.metric("🔢 Est. Tokens", tokens_count)
            
            with st.container(border=True):
                st.markdown(
                    f'<div style="background-color: rgba(15, 23, 42, 0.7); padding: 20px; border-radius: 12px; border: 1px solid rgba(0, 255, 255, 0.4);">{current_text}</div>',
                    unsafe_allow_html=True
                )
                
                st.markdown("---")
                st.markdown("#### ⚡ Quick Prompt Refiner")
                r1, r2, r3 = st.columns(3)
                with r1:
                    if st.button("✂️ Make it Shorter", key=f"short_{opt_key}", use_container_width=True):
                        refine_prompt_text("shorter", opt_key)
                        st.rerun()
                with r2:
                    if st.button("🔍 Add More Details", key=f"det_{opt_key}", use_container_width=True):
                        refine_prompt_text("detailed", opt_key)
                        st.rerun()
                with r3:
                    if st.button("👔 Make it Professional", key=f"prof_{opt_key}", use_container_width=True):
                        refine_prompt_text("professional", opt_key)
                        st.rerun()
                        
                st.markdown("---")
                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st_copy_button(current_text, before_copy_label="📋 Copy Prompt", after_copy_label="✅ Copied!", key=f"copy_{opt_key}")
                with b2:
                    st.download_button(
                        label="💾 Download .txt",
                        data=current_text,
                        file_name=f"{opt_key.replace(' ', '_').lower()}.txt",
                        mime="text/plain",
                        key=f"dl_{opt_key}",
                        use_container_width=True
                    )
                with b3:
                    current_id = st.session_state.current_prompt_id
                    cursor = conn.cursor()
                    cursor.execute("SELECT favorite FROM prompts WHERE id = ?", (current_id,))
                    res = cursor.fetchone()
                    is_fav = bool(res[0]) if res else False
                    
                    fav_label = "⭐ Remove Favorite" if is_fav else "⭐ Save to Favorites"
                    if st.button(fav_label, key=f"fav_{opt_key}", use_container_width=True):
                        if current_id:
                            update_favorite_in_db(current_id, not is_fav)
                            st.rerun()
                with b4:
                    if st.button(f"⚡ Test Run ({opt_key.split()[0]} {opt_key.split()[1]})", key=f"test_{opt_key}", use_container_width=True):
                        test_run_prompt(current_text)
                        st.rerun()

    # --- Live Output Section from Test Run ---
    if st.session_state.test_output:
        st.markdown("### 🤖 Live AI Output (Test Run Result)")
        with st.container(border=True):
            st.markdown(st.session_state.test_output)
            st_copy_button(st.session_state.test_output, before_copy_label="📋 Copy Output", after_copy_label="✅ Output Copied!", key="copy_test_output")
