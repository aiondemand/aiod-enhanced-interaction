from datetime import datetime, timezone


# Now time in UTC stripped of time zone info
# Since MongoDB doesn't store time zone info, we wish to always work with
# time without having a time zone specified (but implicitly always converted first to UTC)
def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)
