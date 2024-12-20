import os

from jinja2 import Environment, FileSystemLoader

if __name__ == "__main__":
    env = Environment(
        loader=FileSystemLoader("."), trim_blocks=True, lstrip_blocks=True
    )
    template = env.get_template("docker-compose.template.yml")

    output = template.render(
        USE_GPU=os.environ.get("USE_GPU", "false"),
        USE_LLM=os.environ.get("USE_LLM", "false"),
    )
    with open("./docker-compose.yml", "w") as f:
        f.write(output)
