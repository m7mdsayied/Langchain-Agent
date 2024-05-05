from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import uuid
model = ChatAnthropic(model="claude-3-haiku-20240307")
prompt = ChatPromptTemplate.from_messages([('placeholder', "{messages}")])
chain = prompt | model
messages = [HumanMessage(content="hi! I'm bob")]
config = {"metadata": {"conversation_id": str(uuid.uuid4())}}
response = chain.invoke({"messages": messages}, config=config)
messages = messages + [response, HumanMessage(content="whats my name")]
response = chain.invoke({"messages": messages}, config=config)
