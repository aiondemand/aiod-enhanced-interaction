import os

from jinja2 import Environment, FileSystemLoader

if __name__ == "__main__":
    USE_GPU = os.environ.get("USE_GPU", "false")
    USE_LLM = os.environ.get("USE_LLM", "false")
    USE_CHATBOT = os.environ.get("USE_CHATBOT", "false")
    USE_MILVUS_LITE = os.environ.get("USE_MILVUS_LITE", "false")

    env = Environment(
        loader=FileSystemLoader("."),
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=True,
    )

    # build Dockerfile
    template = env.get_template("Dockerfile.template.j2")

    output = template.render(
        USE_GPU=USE_GPU,
        USE_CHATBOT=USE_CHATBOT,
    )
    with open("./Dockerfile.final", "w") as f:
        f.write(output)

    # build docker-compose
    template = env.get_template("docker-compose.template.j2")

    output = template.render(
        USE_GPU=USE_GPU,
        USE_LLM=USE_LLM,
        USE_CHATBOT=USE_CHATBOT,
        USE_MILVUS_LITE=USE_MILVUS_LITE,
    )
    with open("./docker-compose.final.yml", "w") as f:
        f.write(output)
