import datetime

import pandas as pd
from sklearn import datasets
from sklearn.metrics import accuracy_score  # Import the accuracy_score function
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
  # Import the metric for classification performance
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase
from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.metrics import DataDriftTable
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import ColumnCategoryMetric
from evidently.metrics import ColumnDistributionMetric
from evidently.metrics import ColumnValuePlot
from evidently.metrics import ColumnQuantileMetric
from evidently.metrics import ColumnCorrelationsMetric
from evidently.metrics import ColumnValueListMetric
from evidently.metrics import ColumnValueRangeMetric
from evidently.metrics import DatasetCorrelationsMetric
from evidently.metrics import ColumnRegExpMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import ColumnMissingValuesMetric
from evidently.metrics import DatasetSummaryMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.metrics import ConflictTargetMetric
from evidently.metrics import ConflictPredictionMetric
from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix
from evidently.metrics import ClassificationQualityByClass
from evidently.metrics import ClassificationClassSeparationPlot
from evidently.metrics import ClassificationProbDistribution
from evidently.metrics import ClassificationRocCurve
from evidently.metrics import ClassificationPRCurve
from evidently.metrics import ClassificationPRTable
from evidently.metrics import ClassificationLiftCurve
from evidently.metrics import ClassificationLiftTable
from evidently.metrics import ClassificationQualityByFeatureTable
from evidently.metrics import ClassificationDummyMetric
from evidently.metrics import RegressionQualityMetric
from evidently.metrics import RegressionPredictedVsActualScatter
from evidently.metrics import RegressionPredictedVsActualPlot
from evidently.metrics import RegressionErrorPlot
from evidently.metrics import RegressionAbsPercentageErrorPlot
from evidently.metrics import RegressionErrorDistribution
from evidently.metrics import RegressionErrorNormality
from evidently.metrics import RegressionTopErrorMetric
from evidently.metrics import RegressionErrorBiasTable
from evidently.metrics import RegressionDummyMetric
# Import the metric for classification performance
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace
from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

from evidently.metrics import ColumnDriftMetric
from evidently.metrics import ColumnSummaryMetric
from evidently.metrics import DatasetDriftMetric
from evidently.metrics import DatasetMissingValuesMetric
from evidently.report import Report
from evidently.test_preset import DataDriftTestPreset
from evidently.test_suite import TestSuite
from evidently.ui.dashboards import CounterAgg
from evidently.ui.dashboards import DashboardPanelCounter
from evidently.ui.dashboards import DashboardPanelPlot
from evidently.ui.dashboards import PanelValue
from evidently.ui.dashboards import PlotType
from evidently.ui.dashboards import ReportFilter
from evidently.ui.remote import RemoteWorkspace

from evidently.ui.workspace import Workspace
from evidently.ui.workspace import WorkspaceBase

adult_data = datasets.fetch_openml(name="adult", version=2, as_frame="auto")
adult = adult_data.frame
import os
root_folder_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
ref_data = pd.read_csv(str(os.path.join(root_folder_path, 'data', 'ref_data.csv')))
prod_data = pd.read_csv(str(os.path.join(root_folder_path, 'data', 'prod_data.csv')))
WORKSPACE = "workspace"
YOUR_PROJECT_NAME = "New Project"
YOUR_PROJECT_DESCRIPTION = "Test project using Adult dataset."




def create_report(i: int):
    y_true = prod_data['target']
    y_pred = prod_data['predicted']

    # Calculate accuracy using sklearn's accuracy_score function
    accuracy = accuracy_score(y_true, y_pred)
    data_drift_report = Report(
        metrics=[
            DatasetDriftMetric(),
            DatasetMissingValuesMetric(),
            ColumnDriftMetric(column_name="target", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="target"),
            ColumnDriftMetric(column_name="predicted", stattest="wasserstein"),
            ColumnSummaryMetric(column_name="predicted"),]

    )
    classification_report = Report(metrics=[
        ClassificationQualityMetric(),
        ClassificationClassBalance(),
        ConflictTargetMetric(),
        ConflictPredictionMetric(),
        ClassificationConfusionMatrix(),
        ClassificationQualityByClass(),
        ClassificationDummyMetric(),
    ])

    classification_report.run(reference_data=ref_data, current_data=prod_data)
    classification_report
    return classification_report


def create_test_suite(i: int):
    data_drift_test_suite = TestSuite(
        tests=[DataDriftTestPreset()],
        timestamp=datetime.datetime.now() + datetime.timedelta(days=i),
    )

    data_drift_test_suite.run(reference_data=ref_data, current_data=prod_data)
    return data_drift_test_suite


def create_project(workspace: WorkspaceBase):
    project = workspace.create_project(YOUR_PROJECT_NAME)
    project.description = YOUR_PROJECT_DESCRIPTION
    project.dashboard.add_panel(
        DashboardPanelCounter(
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            agg=CounterAgg.NONE,
            title="Census Income Dataset (Adult)",
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Model Calls",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetAccuracy",
                field_path=DatasetDriftMetric.fields.current.number_of_rows,
                legend="count",
            ),
            text="count",
            agg=CounterAgg.SUM,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelCounter(
            title="Share of Drifted Features",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            value=PanelValue(
                metric_id="DatasetDriftMetric",
                field_path="share_of_drifted_columns",
                legend="share",
            ),
            text="share",
            agg=CounterAgg.LAST,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="Dataset Quality",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(metric_id="DatasetDriftMetric", field_path="share_of_drifted_columns", legend="Drift Share"),
                PanelValue(
                    metric_id="DatasetMissingValuesMetric",
                    field_path=DatasetMissingValuesMetric.fields.current.share_of_missing_values,
                    legend="Missing Values Share",
                ),
            ],
            plot_type=PlotType.LINE,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="target",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "target"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.dashboard.add_panel(
        DashboardPanelPlot(
            title="prediction",
            filter=ReportFilter(metadata_values={}, tag_values=[]),
            values=[
                PanelValue(
                    metric_id="ColumnDriftMetric",
                    metric_args={"column_name.name": "predicted"},
                    field_path=ColumnDriftMetric.fields.drift_score,
                    legend="Drift Score",
                ),
            ],
            plot_type=PlotType.BAR,
            size=1,
        )
    )
    project.save()
    return project


def create_demo_project(workspace: str):
    ws = Workspace.create(workspace)
    project = create_project(ws)

    for i in range(0, 5):
        report = create_report(i=i)
        ws.add_report(project.id, report)

        test_suite = create_test_suite(i=i)
        ws.add_test_suite(project.id, test_suite)


if __name__ == "__main__":
    create_demo_project(WORKSPACE)
