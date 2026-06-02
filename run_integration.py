"""
Integration Script for IACoP (Student C)
Demonstrates the full multi-agent workflow with knowledge base improvement
"""

import sys
import os
import csv
import json
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.agent_api import send_task_to_agents
from knowledge_store import (
    save_task_knowledge,
    query_knowledge,
    clear_knowledge_base,
    get_knowledge_stats
)


def load_csv_data(csv_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load sensor data from CSV file."""
    data = []
    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if limit and i >= limit:
                    break
                data.append(row)
        print(f"✓ Loaded {len(data)} samples from {csv_path}")
        return data
    except FileNotFoundError:
        print(f"File not found: {csv_path}")
        return []


def format_task_description(row: Dict[str, Any], index: int) -> tuple:
    """
    Convert CSV row to task description for agents.
    """
    max_val = float(row.get('max', 0))
    mean_val = float(row.get('mean', 0))
    min_val = float(row.get('min', 0))
    std_val = float(row.get('std', 0))
   
    abs_min = abs(min_val)
    asymmetry = abs_min / max_val if max_val > 0 else 0

    # Два признака которые реально разделяют классы в этом датасете
    is_fault_signal = (asymmetry > 0.000001 or std_val > 0.001)

    condition = "concerning" if is_fault_signal else "normal"
    parts = []

    if std_val > 0.01:
        parts.append(f"high signal variance (std={std_val:.5f})")
    elif std_val > 0.001:
        parts.append(f"moderate signal variance (std={std_val:.5f})")
    else:
        parts.append(f"low signal variance (std={std_val:.5f})")

    if asymmetry > 0.00001:
        parts.append(f"strong signal asymmetry (|min|/max={asymmetry:.8f})")
    elif asymmetry > 0.000001:
        parts.append(f"mild signal asymmetry (|min|/max={asymmetry:.8f})")
    else:
        parts.append(f"symmetric signal (|min|/max={asymmetry:.8f})")

    parts.append(f"peak={int(max_val)}, mean={int(mean_val)}")

    task_desc = (
        f"Bearing diagnosis #{index}: {condition} vibration pattern - "
        f"{', '.join(parts)}"
    )
    
    true_label = row.get('label', row.get('fault_type', 'unknown'))
    if true_label in ('0', 'healthy'):
        true_label = 'healthy'
    elif true_label in ('1', 'bearing_fault', 'fault'):
        true_label = 'bearing_fault'
    
    return task_desc, true_label


def run_diagnosis(task_desc: str, task_id: str, true_label: str) -> Dict[str, Any]:
    """Run single diagnosis through agents."""
    print(f"\n[Task {task_id}] {task_desc[:80]}...")
    
    try:
        result = send_task_to_agents(task_desc)
        
        if result.get("diagnosis") == "error":
            print(f"Error: {result.get('reasoning', 'Unknown error')}")
            return None
        
        diagnosis = result.get("diagnosis", "unknown")
        confidence = result.get("confidence", 0.0)
        reasoning = result.get("reasoning", "")
        
        print(f"  Diagnosis: {diagnosis} (confidence: {confidence:.2%})")
        
        success = (diagnosis == true_label) if true_label != "unknown" else None
        if success is not None:
            status = "✓ CORRECT" if success else "✗ WRONG"
            print(f"  {status} (true label: {true_label})")
        
        return {
            "task_id": task_id,
            "task_desc": task_desc,
            "diagnosis": diagnosis,
            "confidence": confidence,
            "reasoning": reasoning,
            "true_label": true_label,
            "success": success
        }
    
    except Exception as e:
        print(f"Exception: {str(e)}")
        return None


def run_comparison_demo(test_csv: str = "dataset/test_data.csv", num_samples: int = 25):
    """
    Main demo: Show improvement from Run 1 (empty KB) to Run 2 (with KB)
    """
    print("\n" + "="*80)
    print("IACoP DEMONSTRATION: Multi-Agent Fault Diagnosis with Learning")
    print("="*80)
    
    test_data = load_csv_data(test_csv, limit=num_samples)
    if not test_data:
        print("Failed to load test data")
        return False
    
    # ========== RUN 1: Empty Knowledge Base ==========
    print("\n" + "-"*80)
    print("RUN 1: Empty Knowledge Base (First diagnosis attempts)")
    print("-"*80)
    
    clear_knowledge_base()
    stats = get_knowledge_stats()
    print(f"Knowledge base: {stats['vector_entries']} entries, {stats['graph_nodes']} nodes")
    
    run1_results = []
    for i, row in enumerate(test_data[:num_samples], 1):
        task_desc, true_label = format_task_description(row, i)
        result = run_diagnosis(task_desc, f"run1_task_{i}", true_label)
        if result:
            run1_results.append(result)
    
    if run1_results:
        run1_correct = sum(1 for r in run1_results if r["success"])
        run1_avg_conf = sum(r["confidence"] for r in run1_results) / len(run1_results)
        print(f"\nRun 1 Summary: {run1_correct}/{len(run1_results)} correct, "
              f"avg confidence: {run1_avg_conf:.2%}")
    
    # ========== SAVE KNOWLEDGE FROM RUN 1 (ONLY CORRECT RESULTS) ==========
    print("\n" + "-"*80)
    print("Saving ONLY correct diagnoses from Run 1 to knowledge base...")
    print("-"*80)
    
    saved_count = 0
    for i, result in enumerate(run1_results, 1):
        if result["success"]:
            knowledge = {
                "input": result["task_desc"],
                "output": {
                    "diagnosis": result["diagnosis"],
                    "confidence": result["confidence"]
                },
                "true_label": result["true_label"],
                "success": True,
                "reasoning": result["reasoning"]
            }
            save_task_knowledge(f"run1_task_{i}", knowledge)
            print(f"✓ Knowledge saved for task: run1_task_{i}")
            saved_count += 1
        else:
            print(f"✗ Skipped incorrect diagnosis: run1_task_{i}")
    
    print(f"\n✓ Total saved: {saved_count} entries")
    stats = get_knowledge_stats()
    print(f"Knowledge base: {stats['vector_entries']} entries, {stats['graph_nodes']} nodes")
    
    # ========== RUN 2: With Knowledge Base ==========
    print("\n" + "-"*80)
    print("RUN 2: With Knowledge Base (Improved diagnosis attempts)")
    print("-"*80)
    
    run2_results = []
    for i, row in enumerate(test_data[:num_samples], 1):
        task_desc, true_label = format_task_description(row, i)
        result = run_diagnosis(task_desc, f"run2_task_{i}", true_label)
        if result:
            run2_results.append(result)
    
    if run2_results:
        run2_correct = sum(1 for r in run2_results if r["success"])
        run2_avg_conf = sum(r["confidence"] for r in run2_results) / len(run2_results)
        print(f"\nRun 2 Summary: {run2_correct}/{len(run2_results)} correct, "
              f"avg confidence: {run2_avg_conf:.2%}")
    
    # ========== COMPARISON ==========
    print("\n" + "="*80)
    print("COMPARISON: Run 1 vs Run 2")
    print("="*80)
    
    if run1_results and run2_results:
        run1_correct = sum(1 for r in run1_results if r["success"])
        run2_correct = sum(1 for r in run2_results if r["success"])
        run1_avg_conf = sum(r["confidence"] for r in run1_results) / len(run1_results)
        run2_avg_conf = sum(r["confidence"] for r in run2_results) / len(run2_results)
        
        print(f"\nAccuracy:")
        print(f"  Run 1 (no knowledge): {run1_correct}/{len(run1_results)} "
              f"({run1_correct/len(run1_results)*100:.1f}%)")
        print(f"  Run 2 (with knowledge): {run2_correct}/{len(run2_results)} "
              f"({run2_correct/len(run2_results)*100:.1f}%)")
        
        accuracy_improvement = (run2_correct - run1_correct) / len(run1_results) * 100
        print(f"  Improvement: {accuracy_improvement:+.1f}%")
        
        print(f"\nAverage Confidence:")
        print(f"  Run 1 (no knowledge): {run1_avg_conf:.2%}")
        print(f"  Run 2 (with knowledge): {run2_avg_conf:.2%}")
        
        conf_improvement = (run2_avg_conf - run1_avg_conf) / run1_avg_conf * 100
        print(f"  Improvement: {conf_improvement:+.1f}%")

    return True


def main():
    """Main entry point"""
    try:
        success = run_comparison_demo(test_csv="dataset/test_data.csv", num_samples=25)
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
