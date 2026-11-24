# WhatIf · CausalWorld 推方块

原生 TriFinger 在一次真实试推后，从同一起点想象多种短推，选择第二推。隐藏物理条件由单个粒子向量 `P_k=(方块质量, 地面摩擦系数)` 表示。

- [场景、数据与协议](reports/01-protocol.md)
- [模型、复现命令与结果](reports/02-results.md)

## 复现

服务器的采集环境为 Python 3.10.20，模型训练环境为 Python 3.13.5、NumPy 2.1.3、PyTorch 2.13.0。仿真和拟合可以使用独立解释器。当前实验在 CPU 上完成。已附原始分支数据，可直接从第二步开始。

1. 使用固定版本的上游和无界面物理环境重新采集数据：

   ```bash
   uv venv --python 3.10 .venv
   uv pip install --python .venv/bin/python numpy==1.23.5 scipy==1.10.1 gym==0.23.1 pybullet==3.2.7
   uv pip install --python .venv/bin/python --no-deps 'git+https://github.com/rr-learning/CausalWorld.git@1045277101a92c506db42704da28f6e7227d401b'
   OPENBLAS_NUM_THREADS=1 .venv/bin/python -m whatif.collect --scenes 640 --workers 12 --output work/pairs.npz
   ```

2. 在装有 NumPy 与 PyTorch 的环境中运行：

   ```bash
   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python -m whatif.evaluate --data data/pairs.npz --out results
   python -m whatif.summarize --out results
   ```

场景 10000–10399 用于训练、10400–10479 用于选择检查点和标定观测似然、10480–10639 用于测试。数据文件的数组分别为场景号、两个分叉点的观测、每个动作的实际末端位姿、接触标记、原生重叠分、单一物理向量、各段执行偏差、目标坐标、动作表和试推动作编号。坐标单位米，偏航单位弧度。所有控制器从相同的试推后状态出发。

3. 在新增 240 个独立场景上复现目标对齐的控制器：

   ```bash
   OPENBLAS_NUM_THREADS=1 python -m whatif.goal_control --data data/fresh.npz --out results
   # 若需重采这批数据，使用采集环境
   OPENBLAS_NUM_THREADS=1 .venv/bin/python -m whatif.collect --start 20000 --scenes 240 --workers 12 --output work/fresh.npz
   ```

**结果**：160 个测试场景中，事实更新将第二推终点距离从 23.94 mm 降至 19.87 mm。另 240 个独立场景中，按期望重叠选动作，事实更新将原生成功率从 38.06% 提升至 46.67%。[第三阶段实验报告](reports/03-goal-control.md)
