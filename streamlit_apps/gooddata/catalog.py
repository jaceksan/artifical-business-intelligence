import attr
from pathlib import Path

import streamlit as st

from gooddata.agents.libs.utils import debug_to_file
from gooddata.agents.sdk_wrapper import GoodDataSdkWrapper
from gooddata_sdk import CatalogDeclarativeModel, CatalogDeclarativeAnalytics
from gooddata.agents.libs.rag_langchain import timeit, PRODUCT_NAME
from langchain_core.documents import Document


@attr.s(auto_attribs=True, kw_only=True)
class GoodDataCatalog:
    documents: list[Document]
    ldm: CatalogDeclarativeModel
    adm: CatalogDeclarativeAnalytics


def process_document(path: Path, document_name: str, document: Document, documents: list[Document]) -> None:
    debug_to_file(f"{document_name}.txt", str(document), path)
    documents.append(document)


def create_document(
    workspace_id: str,
    object_type: str,
    main_object: any,
    dependent_object: any = None,
) -> Document:
    content = f"This section describes {PRODUCT_NAME} {object_type}.\n"
    content += f"""This {object_type} has ID "{main_object.id}" and title "{main_object.title}".\n"""
    if dependent_object is not None:
        content += (
            f"""This {object_type} is a part of the standard dataset """ +
            f"""with ID "{dependent_object.id}" and title "{dependent_object.title}".\n"""
        )
    return Document(
        page_content=content, 
        metadata={
            "workspace_id": workspace_id, "object_type": object_type, "id": main_object.id, "title": main_object.title
        }
    )


def generate_description_of_visualization(metrics: list, facts: list, attributes: list) -> str:
    metrics_text = "\n".join([f"- Metric with ID {m['id']}" for m in metrics])
    fact_text = "\n".join([f"- Fact with ID {f['id']} and aggregation function {f['agg_function']}" for f in facts])
    attributes_text = "\n".join([f"- Attribute with ID {a}" for a in attributes])

    result = ""
    if metrics_text is not None and metrics_text != '':
        result += f"""
This visualization contains the following metrics:
{metrics_text}\n"""
    if fact_text is not None and fact_text != '':
        result += f"""
This visualization contains the following facts:
{fact_text}\n"""
    if attributes_text is not None and attributes_text != '':
        result += f"""
This visualization contains the following attributes:
{attributes_text}\n"""

    return result


@st.cache_data
@timeit
def get_gooddata_full_catalog(_gd_sdk: GoodDataSdkWrapper, workspace_id: str, base_path: Path) -> GoodDataCatalog:
    ldm = _gd_sdk.sdk.catalog_workspace_content.get_declarative_ldm(workspace_id)
    adm = _gd_sdk.sdk.catalog_workspace_content.get_declarative_analytics_model(workspace_id)
    metrics = adm.analytics.metrics
    visualizations = adm.analytics.visualization_objects
    dashboards = adm.analytics.analytical_dashboards
    datasets = ldm.ldm.datasets
    date_datasets = ldm.ldm.date_instances
    documents = []
    # DEBUG documents
    for dataset in datasets:
        document = create_document(workspace_id, "dataset", dataset)
        process_document(base_path / "datasets", dataset.id, document, documents)
        for fact in dataset.facts:
            document = create_document(workspace_id, "fact", fact, dataset)
            process_document(base_path / "facts", fact.id, document, documents)
        for attribute in dataset.attributes:
            document = create_document(workspace_id, "attribute", attribute, dataset)
            if len(attribute.labels) > 0:
                label_list = "\n".join([f"""- Label with ID {l.id} has title {l.title}""" for l in attribute.labels])
                document.page_content += f"""
                The attribute also contains the following labels:
                {label_list}
                """
            process_document(base_path / "attributes", attribute.id, document, documents)

    for metric in metrics:
        document = create_document(workspace_id, "metric", metric)
        process_document(base_path / "metrics", metric.id, document, documents)

    for visualization in visualizations:
        document = create_document(workspace_id, "visualization", visualization)
        metrics = []
        attributes = []
        facts = []
        for bucket in visualization.content["buckets"]:
            for item in bucket["items"]:
                if "measure" in item:
                    measure_def = item["measure"]["definition"].get("measureDefinition")
                    if not measure_def:
                        print(f"WARNING: Unknown measure def in visualization: id={visualization.id}")
                    else:
                        if measure_def["item"]["identifier"]["type"] == "fact":
                            facts.append(
                                {"id": measure_def["item"]["identifier"]["id"], "agg_function": measure_def["aggregation"]}
                            )
                        else:
                            metrics.append(
                                {"id": measure_def["item"]["identifier"]["id"]}
                            )
                elif "attribute" in item:
                    attributes.append(item["attribute"]["displayForm"]["identifier"]["id"])
        document.page_content += generate_description_of_visualization(metrics, facts, attributes)
        process_document(base_path / "visualizations", visualization.id, document, documents)

    for dashboard in dashboards:
        # TODO - add details about the dashboard just like above in the visualization
        document = create_document(workspace_id, "dashboard", dashboard)
        process_document(base_path / "dashboards", dashboard.id, document, documents)

    # Date datasets are special. We want to create attributes from them,
    # because that is exactly what is used in report executions.
    for date_dataset in date_datasets:
        document = create_document(workspace_id, "date dataset", date_dataset)
        process_document(base_path / "date_datasets", date_dataset.id, document, documents)
        for granularity in date_dataset.granularities:
            date_attribute_id = f"{date_dataset.id}.{granularity}"
            date_attribute_title = f"{date_dataset.title} {granularity}"
            document = Document(
                page_content=f"""
This {{document}} describes GoodData Phoenix date attribute.
This date attribute has ID "{date_attribute_id}" and title "{date_attribute_title}".
This date attribute is a part of the date dataset with ID "{date_dataset.id}" and title "{date_dataset.title}".
""",
                metadata={
                    "workspace_id": workspace_id,
                    "object_type": "attribute",
                    "id": date_attribute_id,
                    "title": date_attribute_title,
                },
            )
            process_document(base_path / "date_datasets", date_attribute_id, document, documents)

    return GoodDataCatalog(documents=documents, ldm=ldm, adm=adm)
