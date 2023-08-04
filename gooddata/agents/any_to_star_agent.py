from time import time
from enum import Enum
from gooddata_sdk.catalog.data_source.declarative_model.physical_model.pdm import CatalogScanResultPdm
from gooddata.agents.common import GoodDataOpenAICommon
from gooddata.tools import TMP_DIR, create_dir


class AnyToStarResultType(Enum):
    SQL = "SQL statements"
    DBT = "dbt models"


class GoodDataAnyToStarAgent(GoodDataOpenAICommon):

    def scan_data_source(self, data_source_id: str) -> CatalogScanResultPdm:
        # TODO - cache result
        print(f"scan_data_source {data_source_id=} START")
        start = time()
        result = self.gd_sdk.sdk.catalog_data_source.scan_data_source(data_source_id)
        duration = int((time() - start)*1000)
        print(f"scan_data_source {data_source_id=} duration={duration}")
        return result

    @staticmethod
    def generate_list_of_source_tables(pdm: CatalogScanResultPdm) -> str:
        result = f"The source tables are:\n"
        # Only 5 tables for demo purposes, large prompt causes slowness
        # TODO - train a LLM model with PDM model once and utilize it
        i = 1
        for table in pdm.pdm.tables:
            column_list = ", ".join([c.name for c in table.columns])
            result += f"- \"{table.path[-1]}\" with columns {column_list}\n"
            i += 1
            if i >= 5:
                break

        return result

    def process(self, data_source_id: str, result_type: AnyToStarResultType) -> str:
        pdm = self.scan_data_source(data_source_id)

        with open("prompts/any_to_star_schema.txt") as fp:
            prompt = fp.read()
        request = prompt + f"""
Question:
{self.generate_list_of_source_tables(pdm)}
Generate {result_type.value}.
"""
        create_dir(TMP_DIR)
        with open(TMP_DIR / "any_to_star.txt", "w") as fp:
            fp.write(request)

        print(f"OpenAI query START")
        result = self.ask_question(request)
        print(f"OpenAI query END")
        return result
