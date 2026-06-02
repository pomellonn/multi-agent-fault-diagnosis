"""
Manual testing entry point for agent system.
This is not imported by integration; it's just for testing the agents directly.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent_api import send_task_to_agents

def main():
    print("="*80)
    print("IACoP - Agent API Manual Testing")
    print("="*80)
    
    # Test task 1: Bearing fault scenario
    task_1 = """Diagnose the bearing health based on these sensor readings:
    - Vibration frequency: 150Hz
    - Vibration amplitude: 0.8mm
    - Temperature: 75
    - Noise level: 78dB
    
    Note: This machine has shown previous bearing faults at similar frequencies."""
    
    print("\n[Test 1] Bearing Fault Scenario")
    print("-" * 80)
    print(f"Task: {task_1}\n")
    
    result = send_task_to_agents(task_1)
    print("\nResult:")
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Reasoning: {result['reasoning']}")
    
    # Test task 2: Normal operation
    task_2 = """Diagnose the bearing health based on these sensor readings:
    - Vibration frequency: 50Hz
    - Vibration amplitude: 0.05mm
    - Temperature: 45
    - Noise level: 55dB
    
    All readings are within normal ranges."""
    
    print("\n[Test 2] Normal Operation Scenario")
    print("-" * 80)
    print(f"Task: {task_2}\n")
    
    result = send_task_to_agents(task_2)
    print("\nResult:")
    print(f"  Diagnosis: {result['diagnosis']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Reasoning: {result['reasoning']}")
    
    print("\n" + "="*80)
    print("Manual testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
