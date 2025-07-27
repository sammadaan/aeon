import streamlit as st
import time
import numpy as np
import random
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(
    page_title="AEON Consciousness Simulator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===== CORE AEON CLASSES =====

class SensoryModality(Enum):
    VISUAL = 'visual'
    AUDITORY = 'auditory'
    TACTILE = 'tactile'
    OLFACTORY = 'olfactory'
    GUSTATORY = 'gustatory'
    PROPRIOCEPTIVE = "proprioceptive"
    INTEROCEPTIVE = "interoceptive"
    GENERIC_QUALE = "generic_quale"

@dataclass
class SensoryInput:
    modality: SensoryModality
    intensity: float
    location: Optional[np.ndarray] = None
    duration: float = 0.1
    frequency: float = 0.0
    texture: np.ndarray = field(default_factory=lambda: np.array([]))
    semantic_content: str = ""
    temporal_pattern: List[float] = field(default_factory=list)

@dataclass
class BodyState:
    position: np.ndarray
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))
    joint_angles: Dict[str, float] = field(default_factory=dict)
    muscle_tension: Dict[str, float] = field(default_factory=dict)
    energy_level: float = 1.0
    temperature: float = 37.0
    heart_rate: float = 70.0
    breathing_rate: float = 12.0
    pain_signals: Dict[str, float] = field(default_factory=dict)

@dataclass
class MotorCommand:
    action_type: str
    target_location: Optional[np.ndarray] = None
    target_orientation: Optional[np.ndarray] = None
    joint_targets: Dict[str, float] = field(default_factory=dict)
    muscle_targets: Dict[str, float] = field(default_factory=dict)
    force: float = 0.0
    speed: float = 0.0
    precision: float = 0.0
    duration: float = 0.0
    coordination_requirements: List[str] = field(default_factory=list)

class UnconsciousProcessor:
    def __init__(self, processor_id, specialization, processing_capacity):
        self.processor_id = processor_id
        self.specialization = specialization
        self.processing_capacity = processing_capacity

    def process_stimulus(self, stimulus, context):
        processed_data = f"Processed stimulus by {self.processor_id}"
        processing_time = random.uniform(0.01, 0.05)
        urgency = random.random() if 'motor' in self.specialization or 'emotional' in self.specialization else 0.1
        neural_correlates = np.random.rand(10) * np.mean(stimulus) if stimulus.size > 0 else np.random.rand(10)

        result = {
            'processed_data': processed_data,
            'processing_time': processing_time,
            'urgency': urgency,
            'relevance': random.random(),
            'confidence': random.random(),
            'processor_id': self.processor_id,
            'neural_correlates': neural_correlates,
            'processing_result': {},
            'context': context
        }

        if 'motor' in self.specialization:
            result['processing_result']['actions'] = [
                {'action': 'approach', 'confidence': random.random(), 'urgency': random.random()},
                {'action': 'avoid', 'confidence': random.random(), 'urgency': random.random()},
                {'action': 'analyze', 'confidence': random.random(), 'urgency': random.random()}
            ]
            if context.get('stimulus_location') is not None:
                result['processing_result']['location'] = context['stimulus_location'] + np.random.randn(3) * 0.1

        elif 'emotional' in self.specialization:
            result['processing_result']['emotional_vector'] = np.random.rand(3) * 2 - 1
            result['processing_result']['valence'] = np.mean(result['processing_result']['emotional_vector'])

        elif 'visual' in self.specialization:
            result['processing_result']['features'] = np.random.rand(4)
            if context.get('stimulus_location') is not None:
                result['processing_result']['location'] = context['stimulus_location'] + np.random.randn(3) * 0.5

        elif 'memory' in self.specialization:
            result['processing_result']['associations'] = [
                {'memory_result': {'sensory_data_summary': 'something seen before'}},
                {'memory_result': {'emotional_tag': 'pleasant'}}
            ]

        result['competition_score'] = (result['urgency'] * 0.4 +
                                     result['relevance'] * 0.3 +
                                     result['confidence'] * 0.3) * random.uniform(0.8, 1.2)

        return result

class GlobalWorkspace:
    def __init__(self):
        self.current_state = {
            'consciousness_level': 0.0,
            'dominant_processor': None,
            'conscious_content': []
        }
        self.broadcast_threshold = 0.6

    def compete_for_consciousness(self, processor_results: List[Dict]) -> List[Dict]:
        if not processor_results:
            self.current_state = {
                'consciousness_level': 0.0,
                'dominant_processor': None,
                'conscious_content': []
            }
            return []

        sorted_results = sorted(processor_results, key=lambda x: x.get('competition_score', 0), reverse=True)
        conscious_candidates = [
            result for result in sorted_results
            if result.get('competition_score', 0) >= self.broadcast_threshold
        ]

        if not conscious_candidates and sorted_results:
            conscious_candidates = sorted_results[:1]

        if not conscious_candidates:
            self.current_state = {
                'consciousness_level': 0.0,
                'dominant_processor': None,
                'conscious_content': []
            }
            return []

        dominant_processor = conscious_candidates[0]['processor_id']
        consciousness_level = np.mean([c.get('competition_score', 0) for c in conscious_candidates]) if conscious_candidates else 0.0

        self.current_state = {
            'consciousness_level': consciousness_level,
            'dominant_processor': dominant_processor,
            'conscious_content': conscious_candidates
        }

        return conscious_candidates

    def get_conscious_state(self) -> Dict:
        return self.current_state.copy()

class ParallelAEON:
    def __init__(self, num_processors: int = 20):
        self.tick = 0
        self.num_processors = num_processors
        self.processors = []
        
        specializations = ['visual', 'emotional', 'memory', 'motor'] * (num_processors // 4)
        specializations.extend(['generic'] * (num_processors - len(specializations)))

        for i in range(num_processors):
            processor = UnconsciousProcessor(
                processor_id=f"proc_{i}_{specializations[i]}",
                specialization=specializations[i],
                processing_capacity=100
            )
            self.processors.append(processor)

        self.workspace = GlobalWorkspace()
        self.thread_pool = ThreadPoolExecutor(max_workers=num_processors)
        self.conscious_history = []
        self.processing_times = []

    def process_tick_parallel_sync(self, sensory_inputs: List[SensoryInput]) -> Dict:
        """Synchronous version for Streamlit"""
        tick_start = time.time()

        if not sensory_inputs:
            stimulus_vector = np.array([0.0] * 10)
            stimulus_location = None
        else:
            numerical_features = []
            dominant_location = None
            max_intensity = -1.0

            for s_input in sensory_inputs:
                features = [s_input.intensity]
                if isinstance(s_input.texture, np.ndarray) and s_input.texture.size > 0:
                    features.extend(s_input.texture[:5].flatten())
                numerical_features.extend(features)

                if s_input.intensity > max_intensity and s_input.location is not None:
                    max_intensity = s_input.intensity
                    dominant_location = s_input.location

            stimulus_vector = np.array(numerical_features)
            if stimulus_vector.size == 0:
                stimulus_vector = np.array([0.0] * 10)
            
            target_stimulus_size = 20
            if stimulus_vector.size > target_stimulus_size:
                stimulus_vector = stimulus_vector[:target_stimulus_size]
            elif stimulus_vector.size < target_stimulus_size:
                padding_size = target_stimulus_size - stimulus_vector.size
                stimulus_vector = np.pad(stimulus_vector, (0, padding_size), mode='constant')

            stimulus_location = dominant_location

        context = {
            'tick': self.tick,
            'previous_conscious': self.conscious_history[-1]['conscious_content'] if self.conscious_history else [],
            'stimulus_location': stimulus_location,
            'sensory_inputs': sensory_inputs
        }

        # Process synchronously for Streamlit
        processing_results = []
        for processor in self.processors:
            try:
                result = processor.process_stimulus(stimulus_vector, context)
                result['processor_id'] = processor.processor_id
                result['stimulus'] = stimulus_vector.copy()
                processing_results.append(result)
            except Exception as e:
                print(f"Processor {processor.processor_id} failed: {e}")

        conscious_winners = self.workspace.compete_for_consciousness(processing_results)
        conscious_state = self.workspace.get_conscious_state()
        tick_duration = time.time() - tick_start

        tick_result = {
            'tick': self.tick,
            'stimulus': stimulus_vector,
            'total_processors': len(processing_results),
            'conscious_processors': len(conscious_winners),
            'conscious_content': conscious_winners,
            'consciousness_level': conscious_state['consciousness_level'],
            'dominant_processor': conscious_state['dominant_processor'],
            'processing_time': tick_duration,
            'parallel_efficiency': len(processing_results) / len(self.processors) if self.processors else 0.0
        }

        self.conscious_history.append(tick_result)
        self.processing_times.append(tick_duration)
        self.tick += 1

        return tick_result

    def shutdown(self):
        self.thread_pool.shutdown(wait=True)

class RichEnvironment:
    def __init__(self, size: Tuple[int, int, int] = (100, 100, 100)):
        self.size = size
        self.entities = []
        self.time = 0
        self.noise_level = 0.05

        self._add_entity("colorful_object", (50, 50, 10), {"color": "red", "shape": "sphere"}, [SensoryModality.VISUAL])
        self._add_entity("sound_source", (20, 30, 5), {"sound_type": "buzz", "volume": 0.7}, [SensoryModality.AUDITORY])
        self._add_entity("textured_surface", (70, 80, 0), {"texture_type": "rough"}, [SensoryModality.TACTILE])
        self._add_entity("fragrant_flower", (10, 90, 5), {"scent": "floral"}, [SensoryModality.OLFACTORY])
        self._add_entity("sweet_fruit", (60, 10, 10), {"taste": "sweet"}, [SensoryModality.GUSTATORY])

    def _add_entity(self, name: str, location: Tuple[int, int, int], properties: Dict,
                    provides_modalities: List[SensoryModality], is_painful: bool = False):
        self.entities.append({
            "name": name,
            "location": np.array(location),
            "properties": properties,
            "provides": provides_modalities,
            "is_painful": is_painful
        })

    def generate_sensory_input(self, agent_location: np.ndarray, agent_state: Dict) -> List[SensoryInput]:
        inputs = []
        for entity in self.entities:
            distance = np.linalg.norm(agent_location - entity["location"])
            base_intensity = max(0, 1.0 - distance / 50.0)

            for modality in entity["provides"]:
                intensity = base_intensity
                location = entity["location"]
                semantic_content = f"{entity['name']} detected."

                if modality == SensoryModality.VISUAL:
                    color = entity["properties"].get("color", "unknown")
                    shape = entity["properties"].get("shape", "unknown")
                    semantic_content = f"See {color} {shape} ({entity['name']})."
                elif modality == SensoryModality.AUDITORY:
                    volume = entity["properties"].get("volume", 0.5)
                    intensity *= volume
                    semantic_content = f"Hear {entity['properties'].get('sound_type', 'a sound')} ({entity['name']})."

                intensity = np.clip(intensity + random.uniform(-self.noise_level, self.noise_level), 0, 1)

                inputs.append(SensoryInput(
                    modality=modality,
                    intensity=intensity,
                    location=location,
                    semantic_content=semantic_content
                ))

        self.time += 1
        return inputs

class SubjectiveExperienceEngine:
    def __init__(self):
        pass
    
    def process_potential_experience(self, conscious_content, self_model_state, narrative_thread):
        is_conscious = len(conscious_content) > 0
        unity_level = min(1.0, len(conscious_content) / 10.0)
        phenomenal_richness = sum(item.get('urgency', 0) for item in conscious_content) / max(1, len(conscious_content))
        
        subjective_report = "I am experiencing "
        if conscious_content:
            dominant = conscious_content[0]
            processor_type = dominant.get('processor_id', '').split('_')[-1]
            subjective_report += f"{processor_type} processing with intensity {dominant.get('urgency', 0):.2f}."
        else:
            subjective_report += "minimal awareness."
        
        return {
            'is_conscious': is_conscious,
            'unity_level': unity_level,
            'phenomenal_richness': phenomenal_richness,
            'subjective_report': subjective_report
        }

class EmbodiedAEON:
    def __init__(self, environment: RichEnvironment, num_processors: int = 20):
        self.environment = environment
        self.aeon = ParallelAEON(num_processors=num_processors)
        self.qualia_engine = SubjectiveExperienceEngine()
        
        self.body_state = BodyState(
            position=np.array([self.environment.size[0] // 2, self.environment.size[1] // 2, 1], dtype=float)
        )
        
        self.agent_location = self.body_state.position.copy()
        self.agent_state = {
            "emotional_history": deque(maxlen=10), 
            "beliefs": [], 
            "traits": [], 
            "goals": []
        }
        self.narrative_thread = ""
        self.tick_count = 0
        
        self.sensory_history = deque(maxlen=100)
        self.motor_history = deque(maxlen=100)
        self.embodied_memory = deque(maxlen=1000)
        
        self.sensorimotor_integration_level = 0.0
        self.environmental_coupling = 0.0

    def run_simulation_sync(self, ticks: int = 10):
        """Synchronous version for Streamlit"""
        results = []
        
        for i in range(ticks):
            self.tick_count += 1
            
            # Generate sensory input
            sensory_inputs = self.environment.generate_sensory_input(self.agent_location, self.agent_state)
            
            # Process in AEON
            aeon_tick_result = self.aeon.process_tick_parallel_sync(sensory_inputs=sensory_inputs)
            
            # Process subjective experience
            conscious_content = aeon_tick_result['conscious_content']
            self._update_agent_state(conscious_content)
            
            experience_record = self.qualia_engine.process_potential_experience(
                conscious_content=conscious_content,
                self_model_state=self.agent_state,
                narrative_thread=self.narrative_thread
            )
            
            # Update narrative
            if experience_record['is_conscious']:
                self.narrative_thread += " " + experience_record['subjective_report']
                self.narrative_thread = " ".join(self.narrative_thread.split()[-50:])
            
            # Take action
            self._take_action(conscious_content, sensory_inputs)
            
            # Store results
            result = {
                'tick': self.tick_count,
                'sensory_count': len(sensory_inputs),
                'position': self.agent_location.copy(),
                'consciousness_level': experience_record['unity_level'],
                'unity_level': experience_record['unity_level'],
                'phenomenal_richness': experience_record['phenomenal_richness'],
                'conscious_processors': aeon_tick_result['conscious_processors'],
                'total_processors': aeon_tick_result['total_processors'],
                'dominant_processor': aeon_tick_result['dominant_processor'],
                'subjective_report': experience_record['subjective_report']
            }
            results.append(result)
            
            # Small delay
            time.sleep(0.01)
        
        return results

    def _update_agent_state(self, conscious_content: List[Dict]):
        for item in conscious_content:
            processor_id = item.get('processor_id', '')
            processing_result = item.get('processing_result', {})

            if '_emotional' in processor_id and 'emotional_vector' in processing_result:
                self.agent_state['emotional_history'].append(processing_result['emotional_vector'])

            if '_memory' in processor_id and 'associations' in processing_result:
                if processing_result['associations']:
                    new_belief = f"I recall similar experiences."
                    if new_belief not in self.agent_state['beliefs']:
                        self.agent_state['beliefs'].append(new_belief)

    def _take_action(self, conscious_content: List[Dict], sensory_inputs: List[SensoryInput]):
        motor_result = None
        dominant_item_location = None

        if conscious_content:
            dominant_item = conscious_content[0]
            processing_result = dominant_item.get('processing_result', {})
            dominant_loc = processing_result.get('location')
            if dominant_loc is not None and isinstance(dominant_loc, np.ndarray) and dominant_loc.size >= 3:
                dominant_item_location = dominant_loc
                if dominant_item.get('processor_id', '').endswith('_motor'):
                    motor_result = processing_result

        if motor_result and motor_result.get('actions'):
            best_action = max(motor_result['actions'], key=lambda x: x.get('confidence', 0))
            if best_action.get('confidence', 0) > 0.6:
                action = best_action.get('action', 'idle')
                move_speed = 5.0
                direction = np.array([0.0, 0.0, 0.0])

                if dominant_item_location is not None:
                    if action == 'approach':
                        direction = dominant_item_location - self.agent_location
                    elif action == 'avoid':
                        direction = self.agent_location - dominant_item_location

                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)

                new_location = self.agent_location + direction * move_speed
                self.agent_location = np.clip(new_location, [0, 0, 0], np.array(self.environment.size) - 1)
                self.body_state.position = self.agent_location.copy()
                return

        # Fallback behavior
        visual_inputs = [s for s in sensory_inputs if s.modality == SensoryModality.VISUAL]
        if visual_inputs:
            most_interesting = max(visual_inputs, key=lambda x: x.intensity)
            if most_interesting.intensity > 0.5 and most_interesting.location is not None:
                direction = most_interesting.location - self.agent_location
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    new_location = self.agent_location + direction * 3.0
                    self.agent_location = np.clip(new_location, [0, 0, 0], np.array(self.environment.size) - 1)
                    self.body_state.position = self.agent_location.copy()
                return

        # Random exploration
        explore_direction = np.array([random.gauss(0, 1), random.gauss(0, 1), 0])
        new_location = self.agent_location + explore_direction
        self.agent_location = np.clip(new_location, [0, 0, 0], np.array(self.environment.size) - 1)
        self.body_state.position = self.agent_location.copy()

# ===== STREAMLIT APP =====

def main():
    st.title("🧠 AEON Consciousness Simulator")
    st.markdown("*Embodied Artificial Experience & Observation Network*")
    
    # Sidebar controls
    st.sidebar.header("Simulation Parameters")
    
    num_processors = st.sidebar.slider("Number of Processors", 5, 30, 20)
    num_ticks = st.sidebar.slider("Simulation Ticks", 5, 50, 15)
    environment_size = st.sidebar.slider("Environment Size", 50, 200, 100)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Simulation Status")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
    with col2:
        st.header("Controls")
        if st.button("🚀 Start Simulation", type="primary"):
            run_simulation(num_processors, num_ticks, environment_size, status_placeholder, metrics_placeholder)

def run_simulation(num_processors, num_ticks, env_size, status_placeholder, metrics_placeholder):
    """Run the AEON simulation with Streamlit integration"""
    
    try:
        # Initialize
        status_placeholder.info("🔄 Initializing simulation...")
        environment = RichEnvironment(size=(env_size, env_size, env_size))
        aeon = EmbodiedAEON(environment=environment, num_processors=num_processors)
        
        # Progress tracking
        progress_bar = st.progress(0)
        
        status_placeholder.info("🔄 Running simulation...")
        
        # Run simulation
        simulation_data = []
        for tick in range(num_ticks):
            progress_bar.progress((tick + 1) / num_ticks)
            
            # Run one step
            tick_results = aeon.run_simulation_sync(ticks=1)
            if tick_results:
                simulation_data.extend(tick_results)
            
            # Update metrics in real-time
            if len(simulation_data) > 1:
                display_live_metrics(simulation_data, metrics_placeholder)
            
            status_placeholder.success(f"✅ Tick {tick + 1}/{num_ticks} completed")
        
        # Final results
        status_placeholder.success("🎉 Simulation completed!")
        display_final_results(simulation_data)
        
        # Cleanup
        aeon.aeon.shutdown()
        
    except Exception as e:
        status_placeholder.error(f"❌ Simulation failed: {str(e)}")
        st.error(f"Error details: {str(e)}")

def display_live_metrics(data, placeholder):
    """Display real-time metrics"""
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['tick'], y=df['consciousness_level'], name='Consciousness', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['tick'], y=df['unity_level'], name='Unity', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['tick'], y=df['phenomenal_richness'], name='Richness', line=dict(color='red')))
    
    fig.update_layout(title="Real-time Consciousness Metrics", xaxis_title="Tick", yaxis_title="Level", height=400)
    placeholder.plotly_chart(fig, use_container_width=True)

def display_final_results(data):
    """Display comprehensive results"""
    if not data:
        st.warning("No simulation data to display")
        return
        
    df = pd.DataFrame(data)
    
    # Metrics overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Consciousness", f"{df['consciousness_level'].mean():.3f}")
    with col2:
        st.metric("Peak Unity", f"{df['unity_level'].max():.3f}")
    with col3:
        st.metric("Max Richness", f"{df['phenomenal_richness'].max():.3f}")
    with col4:
        st.metric("Total Ticks", len(df))
    
    # Charts
    st.subheader("Consciousness Trajectory")
    
    tab1, tab2, tab3 = st.tabs(["Time Series", "3D Movement", "Statistics"])
    
    with tab1:
        fig = px.line(df, x='tick', y=['consciousness_level', 'unity_level', 'phenomenal_richness'],
                     title="Consciousness Metrics Over Time")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # 3D movement visualization
        if 'position' in df.columns:
            positions = np.array(df['position'].tolist())
            fig = go.Figure(data=go.Scatter3d(
                x=positions[:, 0], y=positions[:, 1], z=positions[:, 2],
                mode='markers+lines',
                marker=dict(size=5, color=df['consciousness_level'], colorscale='Viridis'),
                line=dict(color='darkblue', width=2)
            ))
            fig.update_layout(title="Agent Movement in 3D Space", height=500)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Statistical Summary")
        st.dataframe(df.describe())
        
        # Additional insights
        st.subheader("Consciousness Insights")
        avg_conscious = df['conscious_processors'].mean()
        max_conscious = df['conscious_processors'].max()
        st.write(f"**Average conscious processors:** {avg_conscious:.1f} out of {df['total_processors'].iloc[0]}")
        st.write(f"**Peak conscious processors:** {max_conscious}")
        
        if len(df) > 1:
            consciousness_trend = "increasing" if df['consciousness_level'].corr(df.index) > 0 else "decreasing"
            st.write(f"**Consciousness trend:** {consciousness_trend}")

if __name__ == "__main__":
    main()
