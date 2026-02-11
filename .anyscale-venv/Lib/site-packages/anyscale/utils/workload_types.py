import enum


class Workload(str, enum.Enum):
    JOBS = "jobs"
    SCHEDULED_JOBS = "scheduled_jobs"
    SERVICES = "services"
