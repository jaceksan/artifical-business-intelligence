# Leverage LLM/AI for BI/analytics
Investigating the capabilities of LLM/AI and how we could leverage it for BI/analytics use cases.

All use cases are implemented as Agent classes in [gooddata/agents](gooddata/agents).

Main [gooddata_agents.py](gooddata_agents.py) file embeds these agents into a Streamlit application.

The app is deployed to Streamlit cloud and can be accessed [here](https://artifical-business-intelligence.streamlit.app/).

## Before you start

```bash
cp example.env .env
```

And fill in values for OPENAI_API_KEY and OPENAI_ORGANIZATION.

Then you can run streamlit apps using the following code:

```bash
streamlit run gooddata_agents.py
```

## Agents
So far all agents connect to OpenAI API except GoodData AI Chat, which connects to GoodData AI APIs.

### Chat
Provides ChatGPT-like experience.
Connected to a private space(organization) which is pre-trained by GoodData know-how, it can provide an added value.

### Pandas Data Frame
Leverages create_pandas_dataframe_agent library to answer any questions about data stored in Dataframe.

Users can pick a GoodData insight, which is executed and raw result is displayed as a table.

Then users can ask questions about the data.

### Any to Star Schema
Can generate SQL queries or dbt models.

It is prompted by database metadata (tables, columns) collected by GoodData scan data source functionality.

### Report Agent

Users ask for report. Question is sent to OpenAI with a prompt containing the semantic model (LDM).

OpenAI returns our report definition containing LDM/metrics IDs.

Agent executes the report using the report definition and returns the result as data frame.

You can use [report_execution.ipynb](report_execution.ipynb) notebook as an alternative user interface,
it utilizes the same agent.

### API Agent
OpenAI is prompted with our OpenAPI specification.

Users can ask questions like "list workspaces".

OpenAI answers with what API should be called.

Agent calls the API and returns result as a data frame.

### MAQL Agent
[MAQL](https://www.gooddata.com/docs/cloud/create-metrics/maql/) is a GoodData-specific language for defining metrics.
This agent can generate MAQL from natural language query.

Example:
```
Q: Sum of count of commits where repository name is like "gooddata"

A: SELECT SUM({metric/commit_count}) WHERE {label/repository_name} LIKE "%gooddata%"
```

### RAG
Retrieval Augmented Generation "agent".
Currently it provides only search use case in 3 variants:
- Naive - whole workspace context is sent to LLM
- Vector search - search in vector store
- RAG - search in vector store to reduce context in prompt, call LLM with the prompt

It relies on PRs I created to LangChain fixing both DuckDB and LanceDB LangChain drivers:
- [DuckDB](https://github.com/langchain-ai/langchain/pull/20971)
- [LanceDB](https://github.com/langchain-ai/langchain/pull/21252)
Until these PRs are merged, you need to install LangChain from my fork.

### GoodData AI Chat
GoodData exposes AI APIs via Python SDK.
This agent communicates with GoodData AI Chat API to provide all use cases supported by GoodData AI Chat, e.g.:
- General conversation
- Search for any objects
- Create/extend visualizations

## TODO
1. On-premise LLM as an alternative
2. Procedure to train OpenAI organization and utilize the new model created by OpenAI
