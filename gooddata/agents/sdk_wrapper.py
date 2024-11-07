import os
from typing import Optional

from gooddata_pandas import GoodPandas
from gooddata_sdk import GoodDataSdk


class GoodDataSdkWrapper:
    def __init__(self, profile: Optional[str] = None, timeout: int = 10) -> None:
        self.profile = profile
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
        if self.profile:
            print(f"Connecting to GoodData using profile={self.profile}")
            sdk = GoodDataSdk.create_from_profile(profile=self.profile)
            return sdk
        else:
            kwargs = {}
            if self.override_host:
                kwargs["Host"] = self.override_host
            masked_token = f"{len(self.token[:-4]) * '#'}{self.token[-4:]}"
            print(f"Connecting to GoodData host={self.host} token={masked_token} override_host={self.override_host}")
            sdk = GoodDataSdk.create(host_=self.host, token_=self.token, **kwargs)
            return sdk

    def create_pandas(self) -> GoodPandas:
        return GoodPandas(self.host, self.token, **self.conn_kwargs)

    def wait_for_gooddata_is_up(self) -> None:
        self.sdk.support.wait_till_available(timeout=self.timeout)

    def metrics(self, workspace_id: str) -> list[tuple[str, str]]:
        # TODO - cache the SDK call
        metric_catalog = self.sdk.catalog_workspace_content.get_metrics_catalog(workspace_id=workspace_id)
        return [(metric.id, metric.title) for metric in metric_catalog]

    def facts(self, workspace_id: str) -> list[tuple[str, str]]:
        # TODO - cache the SDK call
        fact_catalog = self.sdk.catalog_workspace_content.get_facts_catalog(workspace_id=workspace_id)
        return [(fact.id, fact.title) for fact in fact_catalog]

    def metrics_string(self, workspace_id: str) -> str:
        return str(self.metrics(workspace_id))

    def attributes(self, workspace_id: str) -> list[tuple[str, str]]:
        # TODO - cache the SDK call
        attribute_catalog = self.sdk.catalog_workspace_content.get_attributes_catalog(workspace_id=workspace_id)
        return [(attr.id, attr.title) for attr in attribute_catalog]

    def attributes_string(self, workspace_id: str) -> str:
        return str(self.attributes(workspace_id))
