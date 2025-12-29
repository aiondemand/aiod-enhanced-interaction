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

    # build Dockerfiles
    cpu_template = env.get_template("Dockerfile.cpu.template.j2")
    gpu_template = env.get_template("Dockerfile.gpu.template.j2")

    cpu_output = cpu_template.render(USE_CHATBOT=USE_CHATBOT)
    gpu_output = gpu_template.render(USE_CHATBOT=USE_CHATBOT)

    with open("./Dockerfile.cpu.final", "w") as f:
        f.write(cpu_output)
    with open("./Dockerfile.gpu.final", "w") as f:
        f.write(gpu_output)

    # build .env.final by concatenating .env.app and rendered .env.template.j2
    env_content = ""

    # Read .env.app if it exists
    if os.path.exists(".env.app"):
        with open(".env.app", "r") as f:
            env_content = f.read()
            # Ensure there's a newline at the end
            if env_content and not env_content.endswith("\n"):
                env_content += "\n"

    # Render .env.template.j2 and append
    template = env.get_template(".env.template.j2")
    env_template_output = template.render(
        USE_GPU=USE_GPU,
        USE_LLM=USE_LLM,
        USE_CHATBOT=USE_CHATBOT,
        USE_MILVUS_LITE=USE_MILVUS_LITE,
    )
    env_content += env_template_output

    # Write the final .env file
    with open("./.env.final", "w") as f:
        f.write(env_content)

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
