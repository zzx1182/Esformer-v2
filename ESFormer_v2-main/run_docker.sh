docker run -it --gpus all \
    -v /home/lqc/OpenPCDet:/workspace/OpenPCDet \
    --shm-size=32g \
    esformer-env:latest


# docker run -it --gpus all \
#     -v /path/to/your/esformer_project/OpenPCDet:/workspace/OpenPCDet \
#     -v /path/to/your/kitti/data:/workspace/OpenPCDet/data/kitti \
#     --shm-size=32g \
#     esformer-env:latest