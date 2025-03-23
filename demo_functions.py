from EvoForge.Agent.task_splitter import TaskDecompositionAgent
from EvoForge import EvoForge  
import time
evoforge_client = EvoForge()
agent = evoforge_client.spawn_setup_agent(session=None)


def task_decomposition(big_task:str)
    agent = TaskDecompositionAgent(model="gpt-4o", session_id="test_task_decomposition1")
    agent.clear_memory()

    agent.rewrite_system_message("You decompose task/multiple tasks into subtasks that can be parallel executed by agents. The current domain is each subtask should be a project setup task such as setting up a github repo or hugging face model.  Plan in very general high level, you must include the full url of the repo/hugging face model.Same as all details that will be used as input for the test. Each task will be picked up by an agent, thus if there are per agent instruction, make sure to mention to them. All task must be parallel executable, thus no sequential tasks")
    response = agent.stream_return_graph_state(big_task)
    return response["plan"]["steps"]


def run_EvoForge_Agent(session:str,task:str):
    agent = evoforge_client.spawn_setup_agent(session)
    agent.clear_memory()
    result = agent.stream_return_graph_state(task)
    time.sleep(20)
    return result

def run_tasks(tasks,n_workers=2):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(run_EvoForge_Agent, task["task_name"], task["task_context"])
            for task in tasks
        ]
        results = [future.result() for future in futures]
    return results


