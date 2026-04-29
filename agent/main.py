from agent_api import send_task_to_agents

def main():
    task_description = (
        "Diagnose the health of the bearing in machine X based on recent vibration and temperature data."
    )
    print(f"Task: {task_description}\n")
    result = send_task_to_agents(task_description)
    print("\nResult")
    for k, v in result.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()