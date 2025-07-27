# Fixed EmbodiedAEON class with missing methods and proper initialization

# First, we need a simple SubjectiveExperienceEngine placeholder
class SubjectiveExperienceEngine:
    def __init__(self):
        pass
    
    def process_potential_experience(self, conscious_content, self_model_state, narrative_thread):
        """Process conscious content into subjective experience"""
        # Simple implementation
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
    """Integrates AEON with an environment and qualia engine"""

    def __init__(self, environment: RichEnvironment, num_processors: int = 20):
        self.environment = environment
        self.aeon = ParallelAEON(num_processors=num_processors)
        self.parallel_aeon = self.aeon  # Alias for compatibility
        self.qualia_engine = SubjectiveExperienceEngine()
        
        # Initialize body state
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
        
        # Initialize history tracking
        self.sensory_history = deque(maxlen=100)
        self.motor_history = deque(maxlen=100)
        self.embodied_memory = deque(maxlen=1000)
        
        # Embodiment metrics
        self.sensorimotor_integration_level = 0.0
        self.environmental_coupling = 0.0

    def perceive_embodied_world(self) -> List[SensoryInput]:
        """Generate sensory inputs from current body state and environment"""
        return self.environment.generate_sensory_input(self.body_state.position, self.agent_state)

    def generate_motor_command(self, sensory_inputs: List[SensoryInput], conscious_content: List[Dict]) -> MotorCommand:
        """Generate motor command based on sensory input and conscious content"""
        # Find motor processor results in conscious content
        motor_results = [item for item in conscious_content if '_motor' in item.get('processor_id', '')]
        
        if motor_results:
            motor_result = motor_results[0]
            actions = motor_result.get('processing_result', {}).get('actions', [])
            if actions:
                best_action = max(actions, key=lambda x: x.get('confidence', 0))
                if best_action.get('confidence', 0) > 0.6:
                    action_type = best_action.get('action', 'move')
                    
                    # Get target location from processing result if available
                    target_location = motor_result.get('processing_result', {}).get('location')
                    if target_location is None:
                        # Default to slight movement forward
                        target_location = self.body_state.position + np.array([5.0, 0.0, 0.0])
                    
                    return MotorCommand(
                        action_type=action_type,
                        target_location=target_location,
                        force=best_action.get('urgency', 0.5),
                        speed=0.5,
                        precision=0.7,
                        duration=0.5
                    )
        
        # Fallback: sensory-driven behavior
        if sensory_inputs:
            strongest_input = max(sensory_inputs, key=lambda x: x.intensity)
            if strongest_input.intensity > 0.3 and strongest_input.location is not None:
                return MotorCommand(
                    action_type='move',
                    target_location=strongest_input.location,
                    force=strongest_input.intensity,
                    speed=0.4,
                    precision=0.6,
                    duration=0.4
                )
        
        # Default: random exploration
        explore_target = self.body_state.position + np.array([
            random.gauss(0, 3), random.gauss(0, 3), 0
        ])
        return MotorCommand(
            action_type='move',
            target_location=explore_target,
            force=0.3,
            speed=0.3,
            precision=0.5,
            duration=0.8
        )

    async def run_simulation(self, ticks: int = 10):
        """Runs the embodied simulation for a number of ticks"""
        print(f"--- Starting Embodied AEON Simulation for {ticks} ticks ---")

        for i in range(ticks):
            self.tick_count += 1
            print(f"\n--- Tick {self.tick_count} ---")

            # 1. Generate sensory input from the environment
            sensory_inputs = self.environment.generate_sensory_input(self.agent_location, self.agent_state)
            print(f"Generated {len(sensory_inputs)} sensory inputs.")

            # 2. AEON processes sensory input in parallel
            aeon_tick_result = await self.aeon.process_tick_parallel(sensory_inputs=sensory_inputs)
            print(f"AEON processed tick. Conscious processors: {aeon_tick_result['conscious_processors']}/{aeon_tick_result['total_processors']}")

            # 3. Subjective Experience Engine processes conscious content
            conscious_content = aeon_tick_result['conscious_content']
            self._update_agent_state(conscious_content)

            experience_record = self.qualia_engine.process_potential_experience(
                 conscious_content=conscious_content,
                 self_model_state=self.agent_state,
                 narrative_thread=self.narrative_thread
            )
            print(f"Subjective Experience: Conscious={experience_record['is_conscious']}, Unity={experience_record['unity_level']:.2f}, Richness={experience_record['phenomenal_richness']:.2f}")

            # 4. Update narrative thread based on conscious experience
            if experience_record['is_conscious']:
                self.narrative_thread += " " + experience_record['subjective_report']
                self.narrative_thread = " ".join(self.narrative_thread.split()[-50:])

            # 5. Agent takes action based on conscious experience
            self._take_action(conscious_content, sensory_inputs)

            # Optional: Add a small delay to simulate real-time ticks
            await asyncio.sleep(0.05)

        print("\n--- Simulation Finished ---")
        self.aeon.shutdown()

    def _update_agent_state(self, conscious_content: List[Dict]):
        """Updates agent's internal state based on conscious processing results"""
        for item in conscious_content:
            processor_id = item.get('processor_id', '')
            processing_result = item.get('processing_result', {})

            if '_emotional' in processor_id and 'emotional_vector' in processing_result:
                self.agent_state['emotional_history'].append(processing_result['emotional_vector'])
                emotional_vec = processing_result.get('emotional_vector')
                if isinstance(emotional_vec, np.ndarray) and emotional_vec.size > 0:
                    avg_emotion = np.mean(emotional_vec)
                    if avg_emotion > 0.7 and 'optimistic' not in self.agent_state['traits']:
                        self.agent_state['traits'].append('optimistic')
                    elif avg_emotion < 0.3 and 'cautious' not in self.agent_state['traits']:
                        self.agent_state['traits'].append('cautious')

            if '_memory' in processor_id and 'associations' in processing_result:
                if processing_result['associations']:
                    assoc_summaries = [assoc['memory_result'].get('sensory_data_summary', 'unknown') for assoc in processing_result['associations'] if 'memory_result' in assoc]
                    if assoc_summaries:
                        new_belief = f"I recall similar experiences involving {assoc_summaries}."
                        if new_belief not in self.agent_state['beliefs']:
                            self.agent_state['beliefs'].append(new_belief)

            if '_motor' in processor_id and 'actions' in processing_result:
                for action_candidate in processing_result.get('actions', []):
                    if action_candidate.get('confidence', 0) > 0.7 and action_candidate.get('urgency', 0) > 0.6:
                        new_goal = f"Goal: {action_candidate.get('action', 'perform action')} related to conscious content."
                        if new_goal not in self.agent_state['goals']:
                            self.agent_state['goals'].append(new_goal)

    def _take_action(self, conscious_content: List[Dict], sensory_inputs: List[SensoryInput]):
        """Agent takes action based on conscious content (simplified movement)"""
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

        # Execute motor action if available
        if motor_result and motor_result.get('actions'):
            best_action = max(motor_result['actions'], key=lambda x: x.get('confidence', 0))
            if best_action.get('confidence', 0) > 0.6:
                action = best_action.get('action', 'idle')
                print(f"Agent taking CONSCIOUS motor action: {action}")

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
                print(f"Agent new location: {self.agent_location}")
                return

        # Fallback to sensory-driven behavior
        visual_inputs = [s for s in sensory_inputs if s.modality == SensoryModality.VISUAL]
        if visual_inputs:
            most_interesting = max(visual_inputs, key=lambda x: x.intensity)
            if most_interesting.intensity > 0.5 and most_interesting.location is not None:
                print(f"Sensory-driven: Approaching visual stimulus")
                direction = most_interesting.location - self.agent_location
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    new_location = self.agent_location + direction * 3.0
                    self.agent_location = np.clip(new_location, [0, 0, 0], np.array(self.environment.size) - 1)
                    self.body_state.position = self.agent_location.copy()
                return

        # Default: small random movement
        explore_direction = np.array([random.gauss(0, 1), random.gauss(0, 1), 0])
        new_location = self.agent_location + explore_direction
        self.agent_location = np.clip(new_location, [0, 0, 0], np.array(self.environment.size) - 1)
        self.body_state.position = self.agent_location.copy()
        print(f"Default: Random exploration to {self.agent_location}")

    async def run_embodied_simulation(self, ticks: int = 20, real_time: bool = True):
        """Run embodied AEON simulation, integrating parallel processing"""
        print("=== Embodied AEON Simulation (Integrated) ===\n")
        return await self.run_simulation(ticks=ticks)

# Fixed test function
async def test_integrated_aeon():
    # Create the environment first
    environment = RichEnvironment(size=(100, 100, 100))
    
    # Now create the EmbodiedAEON with the environment
    aeon = EmbodiedAEON(environment=environment, num_processors=20)
    
    experiences = await aeon.run_embodied_simulation(ticks=15, real_time=True)
    return experiences

if __name__ == "__main__":
    await test_integrated_aeon()
