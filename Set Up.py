pip install -U langchain langsmith


# Make sure your local runtime environment is configured to connect to LangSmith.
export LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
export LANGCHAIN_API_KEY=<your-api-key>

# modify chain to initialize the chain or LLM you want to test.
import langsmith
from langchain import chat_models


# Define your runnable or chain below.
prompt = prompts.ChatPromptTemplate.from_messages(
  [
    ("system", "You are a helpful AI assistant."),
    ("human", "{your_input_key}")
  ]
)
llm = chat_models.ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = prompt | llm | output_parser.StrOutputParser()

client = langsmith.Client()
chain_results = client.run_on_dataset(
    dataset_name="<dataset-name>",
    llm_or_chain_factory=chain,
    project_name="test-dependable-acknowledgment-21",
    concurrency_level=5,
    verbose=True,
)

# Log your first trace

import openai
from langsmith.wrappers import wrap_openai
from langsmith import traceable

# Auto-trace LLM calls in-context
client = wrap_openai(openai.Client())

@traceable # Auto-trace this function
def pipeline(user_input: str):
    result = client.chat.completions.create(
        messages=[{"role": "user", "content": user_input}],
        model="gpt-3.5-turbo"
    )
    return result.choices[0].message.content

pipeline("Hello, world!")
# Out:  Hello there! How can I assist you today?


#Create your first evaluation 

from langsmith import Client
from langsmith.evaluation import evaluate

client = Client()

# Define dataset: these are your test cases
dataset_name = "Sample Dataset"
dataset = client.create_dataset(dataset_name, description="A sample dataset in LangSmith.")
client.create_examples(
    inputs=[
        {"postfix": "to LangSmith"},
        {"postfix": "to Evaluations in LangSmith"},
    ],
    outputs=[
        {"output": "Welcome to LangSmith"},
        {"output": "Welcome to Evaluations in LangSmith"},
    ],
    dataset_id=dataset.id,
)

# Define your evaluator
def exact_match(run, example):
    return {"score": run.outputs["output"] == example.outputs["output"]}

experiment_results = evaluate(
    lambda input: "Welcome " + input['postfix'], # Your AI system goes here
    data=dataset_name, # The data to predict and grade over
    evaluators=[exact_match], # The evaluators to score the results
    experiment_prefix="sample-experiment", # The name of the experiment
    metadata={
      "version": "1.0.0",
      "revision_id": "beta"
    },
)
