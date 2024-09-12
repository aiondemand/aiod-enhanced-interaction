import threading
import time

from app.config import BATCH_INTERVAL_SECONDS


def embedding_thread():
    # TODO in regular intervals check the contents of AIoD
    # Embed documents that haven't been encoded yet

    while True:
        # TODO check AIoD

        time.sleep(BATCH_INTERVAL_SECONDS)


def start_embedding_thread():
    thread = threading.Thread(target=embedding_thread)
    thread.start()
    return thread
