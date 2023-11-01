from typing import Union

import pandas as pd
from pandasai import PandasAI

from gooddata.agents.common import GoodDataOpenAICommon


class GoodDataExplainReportAgent(GoodDataOpenAICommon):

    def ask_pandas_ai_agent(self, df: pd.DataFrame, query: str) -> Union[pd.DataFrame, str, int]:
        pandas_ai = PandasAI(llm=self.get_chat_llm_model())
        return pandas_ai.run(df, prompt=query)

    def process(self, query: str, insight_id: str) -> Union[pd.DataFrame, str, int]:
        # TODO - cache the insight execution?
        df = self.gd_sdk.pandas.data_frames(self.workspace_id).for_insight(insight_id).reset_index()
        print(df)
        return self.ask_pandas_ai_agent(df, query)
