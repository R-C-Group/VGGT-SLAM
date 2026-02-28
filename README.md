 <h1 align="center"> VGGT-SLAM 2.0 测试 </h1>
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
```

## 测试验证
* 主目录下提供了验证的数据`office_loop.zip`:
```bash
unzip office_loop.zip
python3 main.py --image_folder office_loop --max_loops 1 --vis_map
```

若需要3D开放目标检测，采用flag `--run_os`,接下来会提示用户输入文本查询，并且在Viser的地图上绘制检测到的3D bounding box

* 接下来采用手机录制一段视频的建图效果。