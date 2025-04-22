from dotenv import load_dotenv
import os
import yaml
from crewai import Agent, Task, Crew, LLM

load_dotenv() # Read env vars from .env file
print("Env vars loaded..")

# Define LLM model
MODEL_NAME = "gemini/gemini-2.0-flash"
llm = LLM(model=MODEL_NAME)
os.environ['OPENAI_MODEL_NAME'] = MODEL_NAME


# Define file paths for YAML configurations
files = {
    'agents': 'config/agents.yaml',
    'tasks': 'config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']


# Creating Agents
chef = Agent(
  config=agents_config['chef']
)

food_critic = Agent(
  config=agents_config['food_critic']
)

nutritionist = Agent(
  config=agents_config['nutritionist']
)

writer = Agent(
  config=agents_config['writer']
)

# Creating Tasks
recipe_creation = Task(
  config=tasks_config['recipe_creation'],
  agent=chef
)

recipe_critic = Task(
  config=tasks_config['recipe_critic'],
  agent=food_critic
)

nutrition_assessment = Task(
  config=tasks_config['nutrition_assessment'],
  agent=nutritionist
)

recipe_writing = Task(
  config=tasks_config['recipe_writing'],
  agent=writer
)

# Creating Crew
crew = Crew(
  agents=[
    chef,
    food_critic,
    nutritionist,
    writer
  ],
  tasks=[
    recipe_creation,
    recipe_critic,
    nutrition_assessment,
    recipe_writing
  ],
  verbose=True,
  llm=llm
)

# Define needed input variables and create input dict
dish_type="Chicken and Salad"
location="Rio de Janeiro/Brazil"
family_size=4
team_size=2
language="Portuguese"


inputs = {
    'dish_type': dish_type,
    'location': location,
    'family_size': family_size,
    'team_size': team_size,
    'language': language
}

# Run the crew
result = crew.kickoff(
  inputs=inputs
)

print(result)

# Bonus: List expected costs
import pandas as pd

costs = (0.10 * crew.usage_metrics.prompt_tokens + 0.40 * crew.usage_metrics.completion_tokens) / 1_000_000
print("\n\n\n")
print("=====================================================")
print(f"Total costs: ${costs:.4f}")

# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame([crew.usage_metrics.dict()])
df_usage_metrics
