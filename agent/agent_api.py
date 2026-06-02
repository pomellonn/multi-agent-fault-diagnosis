# Agent orchestration API for IACoP.

import sys
import os
import json
import re
from pathlib import Path

# Add current directory to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents.llm_client import build_agent

# Try to import knowledge_store from parent directory
try:
    from knowledge_store import query_knowledge
except ImportError:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from knowledge_store import query_knowledge


def extract_query_from_plan(plan: str) -> str:
    """
    Extract a knowledge query from the Planner's output.
    Looks for relevant keywords and sensor parameters.
    """
    plan_lower = plan.lower()
    
    # Look for specific fault types and parameters
    if "bearing" in plan_lower or "fault" in plan_lower:
        parts = []
        
        if "vibration" in plan_lower:
            parts.append("vibration")
        if re.search(r'frequency|freq|hz', plan_lower):
            parts.append("frequency")
        if re.search(r'amplitude|amp', plan_lower):
            parts.append("amplitude")
        if re.search(r'temperature|temp|heat', plan_lower): 
            parts.append("temperature")
        if "fault" in plan_lower:
            parts.append("fault")
        
        if parts:
            return " ".join(parts)
        return "bearing fault diagnosis"
    
    if "healthy" in plan_lower or "normal" in plan_lower:
        return "normal operation healthy bearing"
    
    # Default: use first meaningful line of plan
    lines = [l.strip() for l in plan.split('\n') if l.strip() and len(l.strip()) > 10]
    if lines:
        return lines[0][:100]
    
    return "fault diagnosis"


def format_retrieved_knowledge(retrieved_results: list) -> str:
    """
    Format retrieved knowledge results for the Reasoner agent.
    ENHANCED: Include quality metrics and guidance for weighting
    """
    if not retrieved_results:
        return "No historical knowledge found. Reason from first principles and be conservative with confidence."
    
    formatted = "Retrieved historical knowledge (use these patterns to weight your diagnosis):\n"
    for i, result in enumerate(retrieved_results, 1):
        text = result.get("text", "")
        score = result.get("score", 0.0)
        
        # Provide guidance on confidence weighting
        if score > 0.85:
            weight = "HIGH - strongly supports your diagnosis"
        elif score > 0.70:
            weight = "MEDIUM - partially relevant"
        else:
            weight = "LOW - tangentially related"
        
        formatted += f"\n{i}. [{weight}] Score: {score:.2%}\n   {text[:150]}...\n"
    
    formatted += "\nIf multiple high-scoring cases agree, increase your confidence. If they disagree, be cautious."
    return formatted


def send_task_to_agents(task_description: str) -> dict:
    """
    Orchestrate multi-agent collaboration for fault diagnosis.
    
    Args:
        task_description: Description of the fault diagnosis task
    
    Returns:
        Dictionary with keys: diagnosis, confidence, reasoning
    """
    try:
        # Instantiate agents
        planner = build_agent("planner")
        retriever = build_agent("retriever")
        reasoner = build_agent("reasoner")

        # Step 1: Planner creates a plan
        planner_prompt = f"""Analyze this bearing condition data and create a diagnostic plan.
        
Data: {task_description}

Create a step-by-step plan for diagnosis, identifying what patterns to look for."""
        
        plan = planner.generate_reply(messages=[{"role": "user", "content": planner_prompt}])
        
        # Step 2: call the Retriever agent to generate query and get knowledge
        retriever_prompt = f"""Based on this diagnostic plan, generate a knowledge query to find similar historical cases:

Plan: {plan}

Generate a concise query to search for similar bearing conditions in historical data."""
        
        query_result = retriever.generate_reply(messages=[{"role": "user", "content": retriever_prompt}])
        
        # Extract query and retrieve knowledge
        query = extract_query_from_plan(query_result)  # Use Retriever's output as the query
        retrieved_results = query_knowledge(query, top_k=3)  # Increased from 2 to 3 for better context
        retrieved_knowledge_str = format_retrieved_knowledge(retrieved_results)

        # Step 3: Reasoner produces final diagnosis with confidence
        # ENHANCED: Include explicit guidance on using retrieved knowledge
        reasoner_prompt = f"""You are diagnosing a bearing condition. Use the plan and historical cases to make your diagnosis.

Data to diagnose:
{task_description}

Diagnostic plan:
{plan}

{retrieved_knowledge_str}

IMPORTANT: If similar cases exist with high relevance scores, weight them heavily in your reasoning.
If cases conflict, be more conservative with confidence. If no similar cases exist, reason from first principles.

Provide your final diagnosis in JSON format:
{{"diagnosis": "healthy" or "bearing_fault", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
"""
        reasoning_json = reasoner.generate_reply(messages=[{"role": "user", "content": reasoner_prompt}])
        
        # Parse and validate JSON response
        try:
            result = json.loads(reasoning_json)
            
            # Validate required keys
            for k in ("diagnosis", "confidence", "reasoning"):
                if k not in result:
                    raise ValueError(f"Missing required key: {k}")
            
            # Ensure confidence is a float between 0 and 1
            if isinstance(result["confidence"], (int, float)):
                result["confidence"] = float(result["confidence"])
                result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            else:
                result["confidence"] = 0.5
            
            # Ensure diagnosis is valid
            if result["diagnosis"] not in ("healthy", "bearing_fault"):
                result["diagnosis"] = "healthy"
            
            return result
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract key information
            return {
                "diagnosis": "bearing_fault" if "fault" in reasoning_json.lower() else "healthy",
                "confidence": 0.5,
                "reasoning": f"Parsed from: {reasoning_json[:200]}..."
            }
            
    except Exception as e:
        return {
            "diagnosis": "error",
            "confidence": 0.0,
            "reasoning": f"Agent execution error: {str(e)}"
        }
