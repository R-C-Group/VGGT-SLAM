 <h1 align="center"> 基于NAVIDA Jeston Thor的VGGT-SLAM 2.0 测试 </h1>

[comment]: <> (  <h2 align="center">PAPER</h2>)
  <h3 align="center">
  <a href="https://github.com/MIT-SPARK/VGGT-SLAM">Github</a>
  | <a href="https://arxiv.org/pdf/2601.19887">Paper</a>
  </h3>
  <div align="center"></div>

## 安装配置

```bash
git clone git@github.com:R-C-Group/VGGT-SLAM.git
conda create -n vggt-slam python=3.11
conda activate vggt-slam
```

* 下载第三方包（这个部分包括了下载所有的第三方包，Perception Encoder, SAM 3，VGGT）：

```bash
chmod +x setup.sh
./setup.sh
# 注意重新修复了bug后再次运行./setup.sh需要：要么清空全部文件，要么注释掉已经完成的脚本
```

* 在`pip install -e ./salad`安装时若存在time out报错，如`WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.org', port=443): Read timed out. (read timeout=15)")': /simple/setuptools/`(PS:本仓库的代码已经改进)

```bash
cd third_party
pip install -e ./salad -i https://pypi.tuna.tsinghua.edu.cn/simple
cd ..
```

* 在`pip install -e ./perception_models -i https://pypi.tuna.tsinghua.edu.cn/simple`安装时报错：

```bash
ERROR: Ignored the following versions that require a different python version: 1.21.2 Requires-Python >=3.7,<3.11; 1.21.3 Requires-Python >=3.7,<3.11; 1.21.4 Requires-Python >=3.7,<3.11; 1.21.5 Requires-Python >=3.7,<3.11; 1.21.6 Requires-Python >=3.7,<3.11; 1.6.2 Requires-Python >=3.7,<3.10; 1.6.3 Requires-Python >=3.7,<3.10; 1.7.0 Requires-Python >=3.7,<3.10; 1.7.1 Requires-Python >=3.7,<3.10; 1.7.2 Requires-Python >=3.7,<3.11; 1.7.3 Requires-Python >=3.7,<3.11; 1.8.0 Requires-Python >=3.8,<3.11; 1.8.0rc1 Requires-Python >=3.8,<3.11; 1.8.0rc2 Requires-Python >=3.8,<3.11; 1.8.0rc3 Requires-Python >=3.8,<3.11; 1.8.0rc4 Requires-Python >=3.8,<3.11; 1.8.1 Requires-Python >=3.8,<3.11
ERROR: Could not find a version that satisfies the requirement decord==0.6.0 (from perception-models) (from versions: none)
ERROR: No matching distribution found for decord==0.6.0
```

* 解决方案：进入目录`third_party/perception_models/requirements.txt`，将`decord==0.6.0`改为`decord2`。（PS：本代码中已经解决）



## 测试验证
* 主目录下提供了验证的数据`office_loop.zip`:
```bash
unzip office_loop.zip
conda activate vggt-slam
python3 main.py --image_folder office_loop --max_loops 1 --vis_map
```

* 首次加载时需要下载dino的权重`https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth`.下载后上传到服务器的目录`/home/kwanwaipang/.cache/torch/hub/checkpoints/`.再次运行命令，程序会检测到本地已有文件，跳过下载步骤。
* 此外，也需要从`https://github.com/serizba/salad/releases`下载`dino_salad.ckpt`到`/home/kwanwaipang/.cache/torch/hub/checkpoints`
* 从`https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt`中下载模型，也是到上述位置（注意VGGT模型较大，有几个G）
* 通过代码`ls -l /home/kwanwaipang/.cache/torch/hub/checkpoints/`再次确认模型已经下载好了。
* 注意检查GPU是否可用`python -c "import torch; print('GPU 是否可用:', torch.cuda.is_available()); print('检测到的 GPU 数量:', torch.cuda.device_count())"`，
* 对于上述输出GPU不可用，大几率是安装时拉取的CUDA版本与Thor显卡驱动不兼容。Thor需要安装:`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`(注意，需要先卸载`pip uninstall torch torchvision -y`)


打开浏览器(输入`http://localhost:8080/`)即可可视化结果：

若需要3D开放目标检测，采用flag `--run_os`,接下来会提示用户输入文本查询，并且在Viser的地图上绘制检测到的3D bounding box


* 接下来采用手机录制一段视频的建图效果。