import os

from jinja2 import Environment, FileSystemLoader

if __name__ == "__main__":
    env = Environment(
        loader=FileSystemLoader("."), trim_blocks=True, lstrip_blocks=True
    )

    # build Dockerfile
    template = env.get_template("Dockerfile.template")

    output = template.render(
        USE_GPU=os.environ.get("USE_GPU", "false"),
    )
    with open("./Dockerfile.final", "w") as f:
        f.write(output)

    # build docker-compose
    template = env.get_template("docker-compose.template.yml")

    output = template.render(
        USE_GPU=os.environ.get("USE_GPU", "false"),
        USE_LLM=os.environ.get("USE_LLM", "false"),
    )
    with open("./docker-compose.final.yml", "w") as f:
        f.write(output)
