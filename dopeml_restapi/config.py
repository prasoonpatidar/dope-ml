from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DopeML API Server"
    B1 = {
        'url': 'edusense-compute-4.andrew.cmu.edu',
        'resnet': {'port': 4001, 'cpus': [1, 2], 'gpus': [], 'memory_in_gbs': 2},
        'ssd': {'port': 4002, 'cpus': [3, 4], 'gpus': [], 'memory_in_gbs': 2},
        'stt': {'port': 4003, 'cpus': [5, 6], 'gpus': [], 'memory_in_gbs': 3},
        'tts': {'port': 4004, 'cpus': [7, 8], 'gpus': [], 'memory_in_gbs': 2},
        'bert': {'port': 4005, 'cpus': [9, 10], 'gpus': [], 'memory_in_gbs': 3},
    }

    B2 = {
        'url': 'edusense-compute-4.andrew.cmu.edu',
        'resnet': {'port': 5001, 'cpus': [11, 12, 13, 14], 'gpus': [], 'memory_in_gbs': 4},
        'ssd': {'port': 5002, 'cpus': [15, 16, 17, 18], 'gpus': [], 'memory_in_gbs': 4},
        'stt': {'port': 5003, 'cpus': [19, 20, 21, 22], 'gpus': [], 'memory_in_gbs': 4},
        'tts': {'port': 5004, 'cpus': [23, 24, 25, 26], 'gpus': [], 'memory_in_gbs': 4},
        'bert': {'port': 5005, 'cpus': [27, 28, 29, 30], 'gpus': [], 'memory_in_gbs': 4},
    }

    B3 = {
        'url': 'edusense-compute-4.andrew.cmu.edu',
        'resnet': {'port': 6001, 'cpus': [31, 32], 'gpus': [0], 'memory_in_gbs': 3},
        'ssd': {'port': 6002, 'cpus': [33, 34], 'gpus': [0], 'memory_in_gbs': 3},
        'stt': {'port': 6003, 'cpus': [33, 36], 'gpus': [0], 'memory_in_gbs': 4.5},
        'tts': {'port': 6004, 'cpus': [37, 38], 'gpus': [0], 'memory_in_gbs': 4},
        'bert': {'port': 6005, 'cpus': [39, 40], 'gpus': [0], 'memory_in_gbs': 4},
    }

    B4 = {
        'url': 'edusense-compute-4.andrew.cmu.edu',
        'resnet': {'port': 7001, 'cpus': [41, 42, 44, 44], 'gpus': [1], 'memory_in_gbs': 6},
        'ssd': {'port': 7002, 'cpus': [45, 46, 47, 48], 'gpus': [1], 'memory_in_gbs': 6},
        'stt': {'port': 7003, 'cpus': [49, 50, 51, 52], 'gpus': [1], 'memory_in_gbs': 6},
        'tts': {'port': 7004, 'cpus': [53, 54, 55, 56], 'gpus': [1], 'memory_in_gbs': 6},
        'bert': {'port': 7005, 'cpus': [57, 58, 59, 60], 'gpus': [1], 'memory_in_gbs': 6},
    }

    # gpuClusterConfigUrl: str = "edusense-compute-4.andrew.cmu.edu"
    # gpuClusterBertPort: int = 3000
    # gpuClusterSSDPort: int = 4000
    # gpuClusterResNetPort: int = 4001
    # gpuClusterSTTPort: int = 6000
    # gpuClusterTTSPort: int = 7000
    #
    # cpuClusterConfig: dict = {"url": "", "port": ""}


settings = Settings()
