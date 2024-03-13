from gooddata.agents.libs.gd_openai import GoodDataOpenAICommon
from gooddata.tools import TMP_DIR, create_dir


class MaqlAgent(GoodDataOpenAICommon):
    def get_open_ai_sys_msg(self) -> str:
        catalog = self.gd_sdk.sdk.catalog_workspace_content.get_full_catalog(self.workspace_id)

        sys_msg = f"""
You are a data engineer writing metrics also known as measures in MAQL language.
You exclusively use the following description of MAQL language.

Metrics always start with keyword SELECT.
You have translate entities specified by natural language with corresponding IDs of entities.
All IDs must be specified as {{<type>/<id>}}, where <type> identifies type of entity and <id> the ID of entity. 

Examples:

Question: Give me count of customers
Answer: SELECT COUNT({{attribute/customer_id}})
   
Question: Give me average price
Answer: SELECT AVG({{fact/price}})

Question: Give me count of flights
Answer: SELECT SUM({{metric/flight_count}})

Now, I teach you how to apply filters. Warning: the entity type for attributes must be here specified as "label". 
Examples:
Question: Sum of flight count where manufacturer is AIRBUS
Answer: SELECT SUM({{metric/flight_count}}) WHERE {{label/manufacturer}} = "AIRBUS"

Question: Sum of flight count where manufacturer is like AIR
Answer: SELECT SUM({{metric/flight_count}}) WHERE {{label/manufacturer}} LIKE "%AIR%"

Now I teach you how to create metrics calculating values for previous period of time.
Examples:
Question: Calculate revenue for previous quarter
Answer: SELECT {{metric/revenue}} FOR Previous({{attribute/quarter}})

Question: Calculate revenue three years before now
Answer: SELECT {{metric/revenue}} FOR Previous({{attribute/year}}, 3)

Here is the list of available entities:
"""
        facts = "Facts:\n" + '\n'.join([f"- Fact with ID {f.id} is described as '{f.title}'" for f in catalog.facts])
        attributes = "Attributes:\n" + '\n'.join(
            [f"- Attribute with ID {a.id} is described as '{a.title}'" for a in catalog.attributes]
        )
        metrics = "Metrics:\n" + '\n'.join(
            [f"- Metric with ID {f.id} is described as '{f.title}'" for f in catalog.metrics]
        )
        sys_msg += f"{facts}\n\n{attributes}\n\n{metrics}"
        create_dir(TMP_DIR)
        with open(TMP_DIR / "maql.txt", "w") as fp:
            fp.write(sys_msg)
        return sys_msg

    @staticmethod
    def get_open_ai_raw_prompt(question: str) -> str:
        return f"""Answer this question: "{question}" """

    def process(self, prompt: str) -> str:
        completion = self.ask_chat_completion(
            system_prompt=self.get_open_ai_sys_msg(),
            user_prompt=self.get_open_ai_raw_prompt(prompt),
        )
        return completion.choices[0].message.content
