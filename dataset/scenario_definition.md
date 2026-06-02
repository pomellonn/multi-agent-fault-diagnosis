Scenario Definition: Bearing Fault Diagnosis with Knowledge Reuse

1. Scenario Overview

This scenario represents an industrial fault diagnosis task using vibration-based sensor data from the UMFDD dataset. The objective is to classify machine condition into one of two categories: healthy or bearing_fault.

The scenario demonstrates the core idea of the IACoP system. Agents improve their performance by reusing previously stored knowledge.

Two runs of the same diagnostic task are compared.

Run 1 (No Knowledge): the knowledge base is empty, and agents rely only on reasoning.

Run 2 (With Knowledge): the knowledge base contains prior examples, so agents can retrieve and reuse knowledge.

The expected outcome is that Run 2 produces a more accurate or more confident diagnosis.

--------------------------------------------------

2. Dataset Description

The dataset used is UMFDD (Mendeley Data).

Classes used:
0 = healthy  
1 = bearing_fault

Dataset preparation:
About 100 samples are selected for each class.

The data is split into two files:

train_data.csv  
50 healthy samples and 50 bearing_fault samples.  
Used as initial knowledge for the knowledge base.

test_data.csv  
50 healthy samples and 50 bearing_fault samples.  
Used for evaluation and demo.

Each sample contains numerical sensor features and a ground-truth label.

--------------------------------------------------

3. Input Specification

Each test sample is converted into a text task description before being passed to the agent system.

Example task:

Task: Diagnose the machine condition based on the following sensor features.

Sample ID: test_001

Sensor features:
feature_1: 0.34  
feature_2: 1.27  
feature_3: 0.89  

Possible diagnoses:
0 = healthy  
1 = bearing_fault  

Return:
diagnosis  
confidence (0 to 1)  
reasoning  

This task is passed to the function:

send_task_to_agents(task_description)

--------------------------------------------------

4. Expected Output

The system returns:

diagnosis: healthy or bearing_fault  
confidence: number between 0 and 1  
reasoning: text explanation  

For each run, the output shows:
- predicted diagnosis
- confidence score
- reasoning
- whether the prediction is correct

--------------------------------------------------

5. Scenario Workflow

The integration script follows these steps:

1. Load one sample from test_data.csv  
2. Build a task description string  
3. Clear the knowledge base using clear_knowledge_base()  
4. Run 1 without knowledge using send_task_to_agents()  
5. Save the result of Run 1  
6. Add knowledge using save_task_knowledge()  
7. Run 2 with knowledge using send_task_to_agents()  
8. Compare the results  

--------------------------------------------------

6. Evaluation Criteria

The scenario is successful if Run 2 performs better than Run 1.

Run 2 is considered better if:
- Run 1 is incorrect and Run 2 is correct, or
- Run 2 has higher confidence than Run 1

Metrics:
- accuracy (correct or incorrect)
- confidence score

--------------------------------------------------

7. Example Result

Ground truth: bearing_fault

Run 1:
Diagnosis: healthy  
Confidence: 0.52  
Result: incorrect  

Run 2:
Diagnosis: bearing_fault  
Confidence: 0.86  
Result: correct  

Conclusion:
Knowledge reuse improved diagnostic performance.

--------------------------------------------------

8. Scenario Goal

The goal of this scenario is to demonstrate that a multi-agent system can improve fault diagnosis performance by retrieving and reusing previously stored knowledge.

This shows that knowledge reuse leads to better results.