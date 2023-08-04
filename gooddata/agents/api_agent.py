import json
import os
from typing import Any, Optional
import pandas as pd
import requests
from openapi_parser.enumeration import OperationMethod
from gooddata.agents.common import GoodDataOpenAICommon
from gooddata.tools import create_dir, TMP_DIR
from openapi_parser.specification import Specification


class ApiAgent(GoodDataOpenAICommon):

    def get_open_ai_sys_msg(self, specification: Specification) -> str:
        sys_msg = f"""
Here is the full list of {self.unique_prefix} APIs:
"""

        for api_path in specification.paths:
            if api_path.url.startswith("/api/v1/entities"):
                for operation in api_path.operations:
                    if operation.method == OperationMethod.GET and operation.summary is not None and operation.summary != "":
                        print(f"------ {api_path.url} -----------")
                        sys_msg += f"""
API path: {api_path.url}
API description: {operation.summary}
"""
        sys_msg += f"""
If you are asked for anything, what relates to the above APIs descriptions, return the corresponding API path.
Example:
Question: list workspaces
Answer: /api/v1/entities/workspaces

Question: Get workspace
Answer: /api/v1/entities/workspaces/{self.workspace_id}

Return only the path, e.g. "/api/v1/entities/workspaces", do not return any other text, no explanation, nothing else than the API path!
"""
        create_dir(TMP_DIR)
        with open(TMP_DIR / "api.txt", "w") as fp:
            fp.write(sys_msg)
        return sys_msg

    @staticmethod
    def get_valid_apis(specification: Specification) -> dict:
        valid_apis = {}
        function_id = 0
        for api_path in specification.paths:
            if api_path.url.startswith("/api/v1/entities"):
                valid_apis[api_path.url] = {}
                for operation in api_path.operations:
                    if operation.method == OperationMethod.GET and operation.summary is not None and operation.summary != "":
                        valid_apis[api_path.url][operation.method] = {
                            "operation": operation,
                            "openapi_function": f"function_{function_id}",
                        }
                        function_id += 1
        return valid_apis

    # TODO - OpenAI limit is 64 functions, we have more APIs ;-)
    def generate_functions(self, specification: Specification) -> list[dict]:
        functions = []
        for api_path, adata in self.get_valid_apis(specification).items():
            for operation_method, odata in adata.items():
                properties = {}
                required = []
                for param in odata['operation'].parameters:
                    properties[param.name] = {
                        # TODO - add support for other types
                        "type": "string",
                        "description": param.description
                    }
                    if param.required:
                        required.append(param.name)
                functions.append(
                    {
                        "name": odata["openapi_function"],
                        "description": odata["operation"].summary,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        }
                    }
                )
        # For debug
        create_dir(TMP_DIR)
        with open(TMP_DIR / "api.json", "w") as fp:
            json.dump(functions, fp)
        return functions

    @staticmethod
    def get_open_ai_raw_prompt(question: str) -> str:
        return f"""Answer this question: "{question}" """

    @staticmethod
    def _prepare_request(path: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            'url': f"{os.environ['GOODDATA_HOST']}{path}",
            'headers': {
                "Content-type": "application/vnd.gooddata.api+json",
                "charset": "utf-8"
            }
        }
        kwargs['headers']['Authorization'] = f'Bearer {os.environ["GOODDATA_TOKEN"]}'
        return kwargs

    @staticmethod
    def _resolve_return_code(response: requests.Response, ok_code: int,
                             url: str, method: str) -> Optional[requests.Response]:
        if response.status_code == ok_code:
            return response
        else:
            print(
                f'{method} to {url} failed - response_code={response.status_code} message={response.text}'
            )
            return None

    def get(self, path: str, ok_code: int = 200) -> Optional[requests.Response]:
        kwargs = self._prepare_request(path)
        response = requests.get(**kwargs)
        return self._resolve_return_code(response, ok_code, kwargs['url'], 'RestApi.get')

    @staticmethod
    def process_entity_data(data: list[dict]) -> list[dict]:
        result = []
        for row in data:
            attributes = {}
            for attr_key, attr_value in row["attributes"].items():
                attributes[attr_key] = attr_value
            result.append({
                "id": row["id"],
                **attributes
            })
        return result

    def process(self, prompt: str, specification: Specification) -> pd.DataFrame:
        completion = self.ask_chat_completion(
            system_prompt=self.get_open_ai_sys_msg(specification),
            user_prompt=self.get_open_ai_raw_prompt(prompt),
        )
        api_path = completion.choices[0].message.content
        if str(api_path).startswith("Answer: "):
            api_path.replace("Answer: ", "")
        api_path = api_path.replace("{workspaceId}", self.workspace_id).replace("{dataSourceId}", "demo")

        print(f"Call {api_path} ...")

        result = self.get(api_path)
        result_dict = json.loads(result.content)["data"]
        print(result_dict)
        data = self.process_entity_data(result_dict)

        return pd.DataFrame.from_dict(data)
