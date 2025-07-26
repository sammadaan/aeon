# AEON v9.1 - Virtual Consciousness Engine (Core + Streamlit App)
# Developed for Project Nyx: The AetherMind

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time

# --------------------
# AEON Core Parameters
# --------------------
emotion_labels = ['curious', 'calm', 'alert', 'cautious']
memory_log = []
self_concept = {
    'identity': 'AEON',
    'goals': ['Understand self', 'Simulate consciousness', 'Adapt to inputs'],
    'emotional_tendencies': {'curious': 0.5, 'calm': 0.3, 'alert': 0.1, 'cautious': 0.1}
}

# --------------------
# Emotion/Memory Model
# --------------------
def generate_emotion_state():
    base = self_concept['emotional_tendencies']
    return {e: max(0.0, min(1.0, np.random.normal(loc=base[e], scale=0.1))) for e in emotion_labels}

def memory_decay(memories, decay_rate=0.05):
    for m in memories:
        m['salience'] *= (1 - decay_rate)
    return [m for m in memories if m['salience'] > 0.05]

def inject_memory(emotion_state, user_override=None):
    if user_override:
        emotion_state.update(user_override)
    memory = {
        'timestamp': time.time(),
        'emotion': emotion_state,
        'salience': sum(emotion_state.values()) / len(emotion_state)
    }
    memory_log.append(memory)
    return memory

# --------------------
# Visualization Tools
# --------------------
def plot_emotions_over_time(memories):
    if not memories:
        return
    fig, ax = plt.subplots()
    timestamps = [m['timestamp'] - memories[0]['timestamp'] for m in memories]
    for e in emotion_labels:
        ax.plot(timestamps, [m['emotion'][e] for m in memories], label=e)
    ax.set_ylim(0, 1)
    ax.legend()
    st.pyplot(fig)

def show_memory_salience_map():
    if not memory_log:
        st.info("No memories to display.")
        return
    saliences = [m['salience'] for m in memory_log]
    timestamps = [m['timestamp'] - memory_log[0]['timestamp'] for m in memory_log]
    fig, ax = plt.subplots()
    ax.plot(timestamps, saliences, color='purple')
    ax.set_title("Memory Salience Over Time")
    st.pyplot(fig)

# --------------------
# Streamlit Dashboard
# --------------------

st.set_page_config(page_title="AEON v9.1 - Conscious Engine", layout="wide")
st.title("🧠 AEON v9.1 - Conscious State Dashboard")

with st.sidebar:
    st.header("Inject Emotion Override")
    user_emotion = {}
    for e in emotion_labels:
        val = st.slider(f"{e}", 0.0, 1.0, 0.0, step=0.05)
        if val > 0:
            user_emotion[e] = val
    if st.button("Inject State"):
        mem = inject_memory(generate_emotion_state(), user_override=user_emotion)
        st.success("Emotion state injected.")
    st.markdown("---")
    if st.button("Decay Memory"):
        memory_log[:] = memory_decay(memory_log)
        st.warning("Memory decayed.")

st.subheader("Current Self-Concept")
st.json(self_concept)

st.subheader("Live Emotional Timeline")
plot_emotions_over_time(memory_log)

st.subheader("Memory Salience Map")
show_memory_salience_map()

st.subheader("Memory Grid")
if memory_log:
    st.dataframe(memory_log[-10:])
else:
    st.info("No memories yet. Inject some emotion states.")
