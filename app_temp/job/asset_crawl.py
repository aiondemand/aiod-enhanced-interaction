import logging
import threading


job_lock = threading.Lock()


# Cleanup expired queries and empty asset collections from MongoDB database
async def crawl_assets_job() -> None:
    if job_lock.acquire(blocking=False):
        try:
            logging.info("[ASSET EXTRACTION] The job has started.")

            await crawl_assets()

            logging.info("[ASSET EXTRACTION] The job has ended.")
        finally:
            job_lock.release()
    else:
        pass


async def crawl_assets() -> None:
    for _ in range(5):
        print("Testing")

    pass
