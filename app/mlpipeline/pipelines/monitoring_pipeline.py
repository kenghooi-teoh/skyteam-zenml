from ..steps.data_fetcher import DataFetcher


class MonitoringPipelineConfig:
    data_fetcher: DataFetcher
    scheduler: ...
    alerter: ...
