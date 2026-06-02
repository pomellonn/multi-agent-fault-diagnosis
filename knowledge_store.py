import chromadb
import networkx as nx
from typing import List, Dict, Any
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer

# ============================================================================
# KNOWLEDGE STORAGE MODULE
# ============================================================================
class KnowledgeStore:
    """Main knowledge storage class combining vector DB and knowledge graph"""
    def __init__(self, persist_directory: str = "./knowledge_base"):
        """Initialize the knowledge store with Chroma and NetworkX"""
        self.persist_directory = persist_directory
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        print("Model loaded.")

        self.collection = self.chroma_client.get_or_create_collection(
            name="task_knowledge",
            metadata={"hnsw:space": "cosine"}
        )
        self.graph = nx.DiGraph()
        self.task_metadata = {}
    def save_task_knowledge(self, task_id: str, knowledge: dict) -> bool:
        """
        Save task knowledge to both vector DB and knowledge graph
        Args:
            task_id: Unique identifier for the task
            knowledge: Dictionary containing task knowledge
                Expected format:
                {
                    "input": str,  # Task description
                    "output": dict,  # Agent's output
                    "true_label": str,  # Correct answer
                    "success": bool  # Whether agent was correct
                }
        Returns:
            bool: True if successful
        """
        try:
            input_text = knowledge.get("input", "")
            output = knowledge.get("output", {})
            true_label = knowledge.get("true_label", "")
            success = knowledge.get("success", False)

            if isinstance(output, dict):
                diagnosis = output.get("diagnosis", "")
                confidence = output.get("confidence", 0.0)
            else:
                diagnosis = str(output)
                confidence = 0.0
            
            searchable_text = f"Task: {input_text} Diagnosis: {diagnosis} True Label: {true_label}"
            
            metadata = {
                "task_id": task_id,
                "true_label": true_label,
                "success": str(success),
                "timestamp": datetime.now().isoformat()
            }
            
            embedding = self.embedding_model.encode([searchable_text]).tolist()

            self.collection.add(
                documents=[searchable_text],
                metadatas=[metadata],
                embeddings=embedding,
                ids=[task_id]
            )
            
            self._extract_graph_triplets(task_id, input_text, diagnosis, true_label, success)
            self.task_metadata[task_id] = {
                "input": input_text,
                "output": output,
                "true_label": true_label,
                "success": success
            }
            print(f"✓ Knowledge saved for task: {task_id}")
            return True
            
        except Exception as e:
            print(f"Error saving knowledge for task {task_id}: {str(e)}")
            return False
    
    def _extract_graph_triplets(self, task_id: str, input_text: str, 
                                diagnosis: str, true_label: str, success: bool):
        """
        Extract simple entity-relation triplets from task knowledge

        This uses simple pattern matching to create basic triplets
        """
        # Create basic triplets from available information
        # Node: task -> relation -> diagnosis
        self.graph.add_node(task_id, type="task")
        self.graph.add_node(diagnosis, type="diagnosis")
        self.graph.add_edge(task_id, diagnosis, relation="produced_diagnosis")
        # Node: task -> relation -> true label
        self.graph.add_node(true_label, type="true_label")
        self.graph.add_edge(task_id, true_label, relation="true_diagnosis")
        if success:
            self.graph.add_edge(diagnosis, true_label, relation="matches")
        else:
            self.graph.add_edge(diagnosis, true_label, relation="differs_from")
        
        import re
        freq_match = re.search(r'frequency\s+(\d+\.?\d*)\s*Hz', input_text, re.IGNORECASE)
        if freq_match:
            freq_value = freq_match.group(1)
            freq_node = f"frequency_{freq_value}Hz"
            self.graph.add_node(freq_node, type="parameter")
            self.graph.add_edge(task_id, freq_node, relation="has_frequency")
            if diagnosis != "unknown":
                self.graph.add_edge(freq_node, diagnosis, relation="indicates")
            
        amp_match = re.search(r'amplitude\s+(\d+\.?\d*)', input_text, re.IGNORECASE)
        if amp_match:
            amp_value = amp_match.group(1)
            amp_node = f"amplitude_{amp_value}"
            self.graph.add_node(amp_node, type="parameter")
            self.graph.add_edge(task_id, amp_node, relation="has_amplitude")
            if diagnosis != "unknown":
                self.graph.add_edge(amp_node, diagnosis, relation="indicates")
        
        # General: true_label -> is_type_of -> class
        if true_label:
            self.graph.add_edge(true_label, "fault_classification", relation="is_type_of")
    
    def query_knowledge(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information
        
        Args:
            query: Search query string
            top_k: Number of results to return
        
        Returns:
            List of dictionaries with keys:
                - "text": knowledge content
                - "score": similarity score (higher = more relevant)
        """
        try:
            if self.collection.count() == 0:
                return []
            query_embedding = self.embedding_model.encode([query]).tolist()

            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=min(top_k, self.collection.count()),
                include=["documents", "distances", "metadatas"]
            )
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    distance = results['distances'][0][i] if results['distances'] else 0
                    score = 1 - distance
                    formatted_results.append({
                        "text": doc,
                        "score": float(score)
                    })
            graph_results = self._query_graph(query)
            if graph_results and formatted_results:
                formatted_results[0]["graph_insights"] = graph_results
            return formatted_results
            
        except Exception as e:
            print(f"Error querying knowledge: {str(e)}")
            return []
    
    def _query_graph(self, query: str) -> List[str]:
        """
        Query the knowledge graph for relevant information
        """
        insights = []
        
        try:
            query_lower = query.lower()
            for node in self.graph.nodes():
                if query_lower in node.lower():
                    neighbors = list(self.graph.neighbors(node))
                    predecessors = list(self.graph.predecessors(node))
                    for neighbor in neighbors:
                        edge_data = self.graph.get_edge_data(node, neighbor)
                        if edge_data:
                            relation = edge_data.get('relation', 'connected_to')
                            insights.append(f"{node} {relation} {neighbor}")
                    for pred in predecessors:
                        edge_data = self.graph.get_edge_data(pred, node)
                        if edge_data:
                            relation = edge_data.get('relation', 'connected_to')
                            insights.append(f"{pred} {relation} {node}")
            
        except Exception as e:
            print(f"Graph query error: {str(e)}")
        return insights[:3]
    
    def clear_knowledge_base(self) -> bool:
        """
        Clear all stored knowledge (both vector DB and graph)
        
        Returns:
            bool: True if successful
        """
        try:
            self.chroma_client.delete_collection("task_knowledge")
            self.collection = self.chroma_client.create_collection(
                name="task_knowledge",
                metadata={"hnsw:space": "cosine"}
            )
            self.graph.clear()
            self.task_metadata = {}
            print("✓ Knowledge base cleared successfully")
            return True
            
        except Exception as e:
            print(f"Error clearing knowledge base: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base
        """
        return {
            "vector_entries": self.collection.count(),
            "graph_nodes": self.graph.number_of_nodes(),
            "graph_edges": self.graph.number_of_edges(),
            "tasks_stored": len(self.task_metadata)
        }

# ============================================================================
# GLOBAL KNOWLEDGE STORE INSTANCE
# ============================================================================
_knowledge_store = KnowledgeStore()

# ============================================================================
# PUBLIC API FUNCTIONS (Must match required signatures)
# ============================================================================
def save_task_knowledge(task_id: str, knowledge: dict) -> bool:
    """
    Save task knowledge to the knowledge base
    
    Args:
        task_id: Unique identifier for the task (e.g., "task_001")
        knowledge: Dictionary containing what to save.
            Format: {
                "input": task description,
                "output": agent's diagnosis result,
                "true_label": correct answer (if known),
                "success": whether agent was correct
            }
    
    Returns:
        bool: True if save was successful
    
    Example:
        >>> save_task_knowledge("task_001", {
        ...     "input": "Vibration frequency 150Hz, amplitude 0.5",
        ...     "output": {"diagnosis": "bearing_fault", "confidence": 0.95},
        ...     "true_label": "bearing_fault",
        ...     "success": True
        ... })
    """
    return _knowledge_store.save_task_knowledge(task_id, knowledge)

def query_knowledge(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    """
    Query the knowledge base for relevant information
    
    Args:
        query: A search string (e.g., "vibration amplitude")
        top_k: Number of results to return (default 2)
    
    Returns:
        List of dictionaries, each containing:
            - "text": the knowledge content (string)
            - "score": similarity score (float, higher = more relevant)
        If no results found, returns empty list []
    
    Example:
        >>> query_knowledge("vibration 150Hz bearing fault", top_k=2)
        [{"text": "Task: ...", "score": 0.95}, ...]
    """
    return _knowledge_store.query_knowledge(query, top_k)

def clear_knowledge_base() -> bool:
    """
    Clear ALL stored knowledge (both vector DB and graph)
    Used for Run 1 (empty knowledge base) vs Run 2 (with knowledge)
    
    Returns:
        bool: True if successful
    
    Example:
        >>> clear_knowledge_base()
    """
    return _knowledge_store.clear_knowledge_base()

def get_knowledge_stats() -> Dict[str, Any]:
    """
    Get current statistics of the knowledge base
    
    Returns:
        Dictionary with statistics about the knowledge base
    """
    return _knowledge_store.get_stats()

def demo_knowledge_system():
    """
    Demonstration script showing save → query → clear functionality
    """
    print("=" * 60)
    print("Knowledge storage system demo")
    print("=" * 60)
    
    print("\n1. Initial state")
    print("-" * 40)
    stats = get_knowledge_stats()
    print(f"Vector entries: {stats['vector_entries']}")
    print(f"Graph nodes: {stats['graph_nodes']}")
    print(f"Graph edges: {stats['graph_edges']}")
    
    result = query_knowledge("vibration test")
    print(f"Query result (should be empty): {result}")
    assert result == [], "Knowledge base should be empty initially"
    print("✓ Empty query returns empty list")
    
    print("\n2. Saving knowledge entries")
    print("-" * 40)
    
    task1_knowledge = {
        "input": "Vibration frequency 150Hz, amplitude 0.5, temperature 85°C",
        "output": {"diagnosis": "bearing_fault", "confidence": 0.95},
        "true_label": "bearing_fault",
        "success": True
    }
    success = save_task_knowledge("task_001", task1_knowledge)
    assert success, "Failed to save task_001"
    
    task2_knowledge = {
        "input": "Vibration frequency 200Hz, amplitude 0.8, irregular pattern detected",
        "output": {"diagnosis": "bearing_fault", "confidence": 0.88},
        "true_label": "bearing_fault",
        "success": True
    }
    success = save_task_knowledge("task_002", task2_knowledge)
    assert success, "Failed to save task_002"
    
    task3_knowledge = {
        "input": "Normal operation, frequency 50Hz, amplitude 0.1, temperature 45°C",
        "output": {"diagnosis": "healthy", "confidence": 0.92},
        "true_label": "healthy",
        "success": True
    }
    success = save_task_knowledge("task_003", task3_knowledge)
    assert success, "Failed to save task_003"
    
    task4_knowledge = {
        "input": "Vibration frequency 180Hz, amplitude 0.6, metallic noise",
        "output": {"diagnosis": "healthy", "confidence": 0.45},
        "true_label": "bearing_fault",
        "success": False
    }
    success = save_task_knowledge("task_004", task4_knowledge)
    assert success, "Failed to save task_004"
    
    stats = get_knowledge_stats()
    print(f"Vector entries after saving: {stats['vector_entries']}")
    print(f"Graph nodes: {stats['graph_nodes']}")
    print(f"Graph edges: {stats['graph_edges']}")
    
    print("\n3. Querying knowledge")
    print("-" * 40)
    
    print("\nQuery 1: 'vibration frequency bearing fault'")
    results = query_knowledge("vibration frequency bearing fault", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['text'][:100]}...")
        if 'graph_insights' in result:
            print(f"  Graph insights: {result['graph_insights']}")
    
    print("\nQuery 2: 'normal operation healthy'")
    results = query_knowledge("normal operation healthy", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['text'][:100]}...")
    
    print("\nQuery 3: 'frequency 180Hz'")
    results = query_knowledge("frequency 180Hz", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text: {result['text'][:100]}...")
    
    print("\n4. Clearing knowledge base")
    print("-" * 40)
    success = clear_knowledge_base()
    assert success, "Failed to clear knowledge base"
    
    stats = get_knowledge_stats()
    print(f"Vector entries after clearing: {stats['vector_entries']}")
    print(f"Graph nodes: {stats['graph_nodes']}")
    
    result = query_knowledge("vibration test")
    print(f"Query result (should be empty): {result}")
    assert result == [], "Knowledge base should be empty after clearing"
    print("✓ Knowledge base successfully cleared")
    
    print("\n5. Testing with sample data (from csv)")
    print("-" * 40)
    
    sample_tasks = [
        {
            "task_id": "run_bearing_128",
            "knowledge": {
                "input": "Bearing vibration analysis: high frequency components, amplitude 0.0095",
                "output": {"diagnosis": "bearing_fault", "confidence": 0.93},
                "true_label": "bearing_fault",
                "success": True
            }
        },
        {
            "task_id": "run_healthy_11",
            "knowledge": {
                "input": "Machine condition monitoring: normal vibration levels, amplitude 0.00069",
                "output": {"diagnosis": "healthy", "confidence": 0.97},
                "true_label": "healthy",
                "success": True
            }
        }
    ]
    for task in sample_tasks:
        save_task_knowledge(task["task_id"], task["knowledge"])
    
    print("\nQuery: 'high frequency vibration fault'")
    results = query_knowledge("high frequency vibration fault", top_k=2)
    for i, result in enumerate(results, 1):
        print(f"Result {i}: Score={result['score']:.4f}")
        print(f"  {result['text'][:80]}...")

    return True

if __name__ == "__main__":
    demo_knowledge_system()
    
    '''print("\n" + "=" * 60)
    print("Integration instructions for students A & C")
    print("=" * 60)
    print("""
Student A (Retriever Agent) should call:
    from knowledge_store import query_knowledge
    results = query_knowledge(query_string, top_k=2)

Student C (Integration Script) should call:
    from knowledge_store import save_task_knowledge, clear_knowledge_base
    save_task_knowledge(task_id, knowledge_dict)
    clear_knowledge_base()

Knowledge dict format expected by save_task_knowledge:
    {
        "input": "task description string",
        "output": {"diagnosis": "diagnosis_string", "confidence": float},
        "true_label": "correct_label_string",
        "success": boolean
    }
    """)'''