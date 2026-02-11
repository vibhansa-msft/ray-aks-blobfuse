from typing import Optional

from anyscale.api_utils.exceptions.job_errors import NoJobRunError
from anyscale.sdk.anyscale_client.api.default_api import DefaultApi as BaseApi
from anyscale.sdk.anyscale_client.models.production_job import ProductionJob


def _get_job_run_id(
    base_api: BaseApi,
    *,
    job_id: Optional[str] = None,
    job_run_id: Optional[str] = None,
) -> str:
    assert bool(job_id) != bool(
        job_run_id
    ), "Exactly one of `job_id` or `job_run_id` must be provided."
    if job_id:
        prod_job: ProductionJob = base_api.get_production_job(job_id).result
        if not prod_job.last_job_run_id:
            raise NoJobRunError(f"Production job {job_id} has no job runs.")
        job_run_id = prod_job.last_job_run_id
    return job_run_id  # type: ignore
