import os
from subprocess import check_output

from timemachine.parallel.grpc.service_pb2 import StatusResponse


def get_gpu_count() -> int:
    # Expected to return a line delimited summary of each GPU
    output = check_output(["nvidia-smi", "-L"])
    gpu_list = [x for x in output.split(b"\n") if len(x)]

    # Respect CUDA_VISIBLE_DEVICES in determining GPU count
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        gpu_list = [gpu_list[i] for i in map(int, visible_devices.split(","))]

    return len(gpu_list)


def get_worker_status() -> StatusResponse:
    try:
        with open("/proc/driver/nvidia/version") as ifs:
            nvidia_driver = ifs.read().strip()
    except FileNotFoundError:
        nvidia_driver = ""
    try:
        git_sha = check_output(["git", "rev-parse", "HEAD"]).strip()
    except FileNotFoundError:
        git_sha = ""
    return StatusResponse(
        nvidia_driver=nvidia_driver,
        git_sha=git_sha,
    )
