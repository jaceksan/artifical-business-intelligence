from enum import Enum


class GoodDataAgent(Enum):
    CHAT = "Chat"
    EXPLAIN_DATA = "Explain data"
    ANY_TO_STAR = "Any to Star Model"
    REPORT_EXECUTOR = "Report executor"
    API_EXECUTOR = "API executor"
    MAQL_GENERATOR = "MAQL generator"


class ChartType(Enum):
    TABLE = "Table"
    BAR_CHART = "Bar chart"
    LINE_CHART = "Line chart"
