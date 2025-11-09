import logging

# TODO resolve
# class HealthCheckFilter(logging.Filter):
#     def filter(self, record: logging.LogRecord) -> bool:
#         # Drop logs for the health endpoint
#         return "/health" not in record.getMessage()


def setup_logger():
    format_string = "%(asctime)s [%(levelname)s] %(name)s - %(message)s (%(filename)s:%(lineno)d)"
    logging.basicConfig(
        level=logging.INFO,
        format=format_string,
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


# logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
