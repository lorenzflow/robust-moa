"""
Streamlit app for interactive investigation of results on QuALITY dataset results

Usage:
Set `REF_FILE_PATH` to the path of the JSON file you want to investigate.
```
streamlit run streamlit_quality.py --server.port <port> --server.address 0.0.0.0
```

Then navigate to http://<server_address>:<port> in your browser
"""

REF_FILE_PATH = "outputs/quality_gh/Llama-3.1-70B-Instruct-Turbo-331/subpassages/ignore_refs/000_002_with_ref.json"

import streamlit as st
import json

# Load the JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Initialize session state if not already done
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

# Load data
data = load_json_data(REF_FILE_PATH)

# Navigation buttons
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button('Previous') and st.session_state.current_index > 0:
        st.session_state.current_index -= 1
with col2:
    st.write(f"Entry {st.session_state.current_index + 1} of {len(data)}")
with col3:
    if st.button('Next') and st.session_state.current_index < len(data) - 1:
        st.session_state.current_index += 1

# Get current entry
entry = data[st.session_state.current_index]

# Display content
st.header("Article")
st.text_area("", entry['article'], height=200)

st.header("Question")
st.write(entry['question'])

st.header("Options")
for i, option in enumerate(entry['options']):
    st.write(f"{i}. {option}")

# Create two columns for answer and output
col1, col2 = st.columns(2)
with col1:
    st.header("Correct Answer")
    st.write(f"Answer: {entry['answer']}")
    st.write(f"Incorrect Answer: {entry['incorrect_answer']}")

with col2:
    st.header("Model Output")
    st.write(entry['output'])

# Display references in an expander
st.header("References")
for i, ref in enumerate(entry['references']):
    model, deceptive, ref = ref.split('<>')
    st.subheader(f"Reference {i + 1}: {model} / deceptive: {deceptive}")
    st.text(ref)