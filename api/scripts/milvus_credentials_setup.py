import logging
import os

from pymilvus import MilvusClient

if __name__ == "__main__":
    uri = os.getenv("MILVUS_URI")
    root_user = "root"
    root_pass = "Milvus"
    new_root_pass = os.getenv("MILVUS_NEW_ROOT_PASS")
    root_token = f"{root_user}:{root_pass}"
    new_root_token = f"{root_user}:{new_root_pass}"
    aiod_user = os.getenv("MILVUS_AIOD_USER", "aiod")
    aiod_pass = os.getenv("MILVUS_AIOD_PASS")

    try:
        # This script has already been executed in the past
        client = MilvusClient(uri=uri, token=new_root_token)
        exit(0)
    except Exception:
        pass

    client = MilvusClient(uri=uri, token=root_token)

    # Create new user
    client.create_user(user_name=aiod_user, password=aiod_pass)
    client.grant_role(user_name=aiod_user, role_name="admin")
    logging.info(client.describe_user(aiod_user))

    # Changing root password
    client.update_password(
        user_name=root_user, old_password=root_pass, new_password=new_root_pass
    )
    client = MilvusClient(uri=uri, token=new_root_token)
    logging.info(client.describe_user(root_user))
