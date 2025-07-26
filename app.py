# app.py

import streamlit as st
from aeon import AEON

st.set_page_config(page_title="AEON State Visualizer", layout="centered")
st.title("🌌 AEON: Virtual Consciousness Engine")

# Initialize AEON
aeon = AEON()

# Sidebar input
st.sidebar.header("Stimuli Input")
stimuli = st.sidebar.multiselect(
    "Select environmental stimuli to feed into AEON:",
    options=["light", "movement", "dark", "stillness"],
    default=["light", "movement"]
)

# Apply stimuli and decay
if st.sidebar.button("Update AEON"):
    aeon.update_states(stimuli)
    aeon.decay_memory_salience()

# Show states
st.subheader("🧠 Emotional States")
aeon.visualize_states()
st.pyplot()

# Show memory salience
st.subheader("🧬 Memory Salience")
aeon.visualize_memory_salience()
st.pyplot()

# Show self-concept drift
st.subheader("🔁 Self-Concept Drift Matrix")
aeon.visualize_self_concept_drift()
st.pyplot()

# Footer
st.markdown("---")
st.caption("Project Nyx: The AetherMind – AEON Core v8.11")

