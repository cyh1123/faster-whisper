# 获取 conda 的安装路径
CONDA_BASE=$(conda info --base)

# 初始化 conda
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate agent

export LD_LIBRARY_PATH=$(python -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))')


CUDA_VISIBLE_DEVICES=7 python whisper_streaming/whisper_online_server_websocket.py --model=models/faster-whisper-large-v3 --lan=zh