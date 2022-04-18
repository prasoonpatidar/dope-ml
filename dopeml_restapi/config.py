from pydantic import BaseSettings


class Settings(BaseSettings):
    app_name: str = "DopeML API Server"
    gpuClusterConfigUrl: str = "edusense-compute-4.andrew.cmu.edu"
    gpuClusterBertPort: int = 3000
    gpuClusterSSDPort: int = 4000
    gpuClusterResNetPort: int = 5000
    gpuClusterSTTPort: int = 6000
    gpuClusterTTSPort: int = 7000

    cpuClusterConfig: dict = {"url": "", "port": ""}


settings = Settings()
