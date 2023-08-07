from typing import Union

import pandas as pd
from langchain.agents import create_pandas_dataframe_agent
from pandasai import PandasAI

from gooddata.agents.common import GoodDataOpenAICommon


class GoodDataExplainReportAgent(GoodDataOpenAICommon):
    def ask_create_pd_agent(self, df: pd.DataFrame, query: str) -> str:
        agent = create_pandas_dataframe_agent(
            self.get_openai(),
            df,
            verbose=True,
            max_iterations=10,
        )
        # Run the prompt through the agent and capture the response.
        # Enforce using python_repl_ast tool, otherwise the library reports errors like this:
        #   Observation: I will use the `groupby` function to ... is not a valid tool, try another one.
        context = "Context: Always use tool python_repl_ast!\n\n"
        final_query = context + "Question: \n" + query
        print(final_query)
        response = agent.run(final_query)
        # Return the response converted to a string.
        return str(response)

    def ask_pandas_ai_agent(self, df: pd.DataFrame, query: str) -> Union[pd.DataFrame, str, int]:
        pandas_ai = PandasAI(self.get_openai())
        return pandas_ai.run(df, prompt=query)

    def process(self, query: str, insight_id: str) -> Union[pd.DataFrame, str, int]:
        # TODO - cache the insight execution?
        df = self.gd_sdk.pandas.data_frames(self.workspace_id).for_insight(insight_id).reset_index()
        print(df)
        return self.ask_pandas_ai_agent(df, query)
