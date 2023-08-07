import os
from typing import Optional

from gooddata_pandas import GoodPandas
from gooddata_sdk import GoodDataSdk


class GoodDataSdkWrapper:
    def __init__(self, timeout: int = 600) -> None:
        self.timeout = timeout
        self.sdk = self.create_sdk()
        self.pandas = self.create_pandas()
        self.wait_for_gooddata_is_up()

    @property
    def host(self) -> str:
        return os.getenv("GOODDATA_HOST", "http://localhost:3000")

    @property
    def token(self) -> str:
        return os.getenv("GOODDATA_TOKEN", "YWRtaW46Ym9vdHN0cmFwOmFkbWluMTIz")

    @property
    def override_host(self) -> Optional[str]:
        return os.getenv("GOODDATA_OVERRIDE_HOST")

    @property
    def conn_kwargs(self):
        kwargs = {}
        if self.override_host:
            kwargs["Host"] = self.override_host
        return kwargs

    def create_sdk(self) -> GoodDataSdk:
        sdk = GoodDataSdk.create(host_=self.host, token_=self.token, **self.conn_kwargs)
        return sdk

    def create_pandas(self) -> GoodPandas:
        return GoodPandas(self.host, self.token, **self.conn_kwargs)

    def wait_for_gooddata_is_up(self) -> None:
        self.sdk.support.wait_till_available(timeout=self.timeout)

    def metrics(self, workspace_id: str) -> list[list[str, str]]:
        # TODO - cache the SDK call
        return [
            [metric.id, metric.title]
            for metric in self.sdk.catalog_workspace_content.get_metrics_catalog(workspace_id=workspace_id)
        ]

    def metrics_string(self, workspace_id: str) -> str:
        return str(self.metrics(workspace_id))

    def attributes(self, workspace_id: str) -> list[list[str, str]]:
        # TODO - cache the SDK call
        return [
            [attr.id, attr.title]
            for attr in self.sdk.catalog_workspace_content.get_attributes_catalog(workspace_id=workspace_id)
        ]

    def attributes_string(self, workspace_id: str) -> str:
        return str(self.attributes(workspace_id))
