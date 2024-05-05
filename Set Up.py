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

