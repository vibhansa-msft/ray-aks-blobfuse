from datetime import date

from anyscale._private.sdk.base_sdk import BaseSDK
from anyscale.aggregated_instance_usage.models import DownloadCSVFilters


class PrivateAggregatedInstanceUsageSDK(BaseSDK):
    def download_csv(self, filters: DownloadCSVFilters) -> str:
        cloud_id = (
            self.client.get_cloud_id(cloud_name=filters.cloud, compute_config_id=None)
            if filters.cloud
            else None
        )

        project_id = (
            self.client.get_project_id(parent_cloud_id=cloud_id, name=filters.project)
            if filters.project
            else None
        )

        return self.client.download_aggregated_instance_usage_csv(
            start_date=date.fromisoformat(filters.start_date),
            end_date=date.fromisoformat(filters.end_date),
            cloud_id=cloud_id,
            project_id=project_id,
            directory=filters.directory,
            hide_progress_bar=filters.hide_progress_bar
            if filters.hide_progress_bar
            else False,
        )
