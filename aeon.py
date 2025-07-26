# aeon.py

import numpy as np
import matplotlib.pyplot as plt

class AEON:
    def __init__(self):
        self.states = {
            'curious': 0.0,
            'calm': 0.0,
            'alert': 0.0,
            'cautious': 0.0
        }
        self.memory_salience = {
            'stillness': 2.2,
            'light': 3.1,
            'movement': 1.5,
            'dark': 0.8
        }
        self.self_concept_drift = {
            'calm-stillness_approach': 1,
            'calm-stillness_avoid': 1,
            'calm-stillness_analyze': 1,
            'calm-light_avoid': 1,
            'calm-light_analyze': 3,
            'calm-movement_approach': 2,
            'calm-dark_approach': 1
        }

    def update_states(self, stimuli):
        # Basic rules for state activation based on stimulus
        for stimulus in stimuli:
            if stimulus == 'light':
                self.states['curious'] += 0.4
                self.states['calm'] += 0.3
            elif stimulus == 'movement':
                self.states['alert'] += 0.5
            elif stimulus == 'dark':
                self.states['cautious'] += 0.4
            elif stimulus == 'stillness':
                self.states['calm'] += 0.4

        # Normalize states
        total = sum(self.states.values())
        if total > 0:
            for k in self.states:
                self.states[k] /= total

    def decay_memory_salience(self):
        for k in self.memory_salience:
            self.memory_salience[k] *= 0.9

    def visualize_states(self):
        labels = list(self.states.keys())
        values = list(self.states.values())
        plt.figure()
        for i in range(len(labels)):
            plt.plot([0, 10], [values[i], values[i]], label=labels[i], linewidth=2)
        plt.ylim(0, 1)
        plt.legend()
        plt.show()

    def visualize_memory_salience(self):
        plt.figure()
        keys = list(self.memory_salience.keys())
        values = list(self.memory_salience.values())
        plt.bar(keys, values, color='purple')
        plt.title("Memory Salience (Decayed)")
        plt.xlabel("Perception")
        plt.ylabel("Weight")
        plt.show()

    def visualize_self_concept_drift(self):
        plt.figure()
        labels = list(self.self_concept_drift.keys())
        values = list(self.self_concept_drift.values())
        plt.barh(labels, values, color='orange')
        plt.title("Self-Concept Drift Matrix")
        plt.xlabel("Frequency")
        plt.tight_layout()
        plt.show()


# Optional: For testing AEON locally
if __name__ == "__main__":
    aeon = AEON()
    aeon.update_states(['light', 'movement'])
    aeon.decay_memory_salience()
    aeon.visualize_states()
    aeon.visualize_memory_salience()
    aeon.visualize_self_concept_drift()
