## Docker 환경 설정

[pytorch/pytorch Docker hub](https://hub.docker.com/r/pytorch/pytorch)

```docker pull pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime```

```pip install requirments.txt```

```apt-get update && apt-get install -y libglib2.0-0 libsm6 libxext6 libgl1-mesa-glx```

```docker commit {CONTAINER ID} afclip```

```sudo docker run -it --rm -v /mnt/c/Users/Ajou/Desktop/capston:/workspace afclip```
