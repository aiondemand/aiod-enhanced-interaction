import os
from time import sleep

from pymilvus import MilvusClient

if __name__ == "__main__":
    start_msg = "============ Milvus credentials setup INITIALIZED ============"
    fin_msg = "============ Milvus credentials setup COMPLETED ============"

    sleep(10)  # Headstart for Milvus to fully initialize

    print(start_msg)
    uri = os.getenv("MILVUS__URI")
    new_root_pass = os.getenv("MILVUS_NEW_ROOT_PASS")
    new_root_token = f"root:{new_root_pass}"
    aiod_user = os.getenv("MILVUS__USER", "aiod")
    aiod_pass = os.getenv("MILVUS__PASS")

    try:
        client = MilvusClient(uri=uri, token=new_root_token)
        print("The script has already been executed in the past")
        print(fin_msg)
        client.close()
        exit(0)
    except Exception:
        pass

    client = MilvusClient(uri=uri, token="root:Milvus")

    client.create_user(user_name=aiod_user, password=aiod_pass)
    client.grant_role(user_name=aiod_user, role_name="admin")
    print("CREATED NEW USER:", client.describe_user(aiod_user))

    # Changing root password
    client.update_password(
        user_name="root", old_password="Milvus", new_password=new_root_pass
    )
    client = MilvusClient(uri=uri, token=new_root_token)
    print("CHANGED ROOT USER CREDENTIALS:", client.describe_user("root"))

    client.close()
    print(fin_msg)
    exit(0)
