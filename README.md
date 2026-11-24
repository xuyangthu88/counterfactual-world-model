# WhatIf · 基于事实交互的反事实世界模型

原生 TriFinger 在一次真实试推后，从同一起点想象多种短推，选择第二推。隐藏物理条件由单个粒子向量 `P_k=(方块质量, 地面摩擦系数)` 表示。

- [场景、数据与协议](reports/01-protocol.md)
- [模型、复现命令与结果](reports/02-results.md)

## ICML 2027 单种子实验

训练随机种子固定为 11，两个比较方法为状态版事实残差复用、稠密关系物理后验。首先检验只用单块和双块训练时向三块的迁移，然后给各方法增加相同的 320 个三块场景，再将冻结模型部署到 CausalWorld 原生 Pushing。报告同时给出效果差分布、真实终点和策略所选动作。

- [前人研究的场景、反事实真值与控制设置，分批方案](reports/pilot/00-survey.md)
- [同起点反事实预测，两个基线和校正公式](reports/pilot/01-prediction.md)
- [固定与自适应首推、第二推动作排序](reports/pilot/02-control.md)
- [原生 Pushing，P1/P2 质量干预下的五推](reports/pilot/03-native.md)
- [独立批次的重新定向与终止](reports/pilot/04-steering.md)
- [以两推终局价值选择第一推](reports/pilot/05-first-value.md)
- [分支面板](data/pilot) · [权重、逐场景决策与结果](results/pilot)

复现使用下文的 CausalWorld 仿真环境，以及含 PyTorch、NumPy、scikit-learn 1.6.1 与 joblib 的训练环境。已有分叉数据和权重时可直接执行评测。

```bash
# 生成同一训练种子的新分支与检查点
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m whatif.pilot_scenes
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m whatif.pilot_scenes --value --out data/pilot/value
python -m whatif.pilot_train --phase zero --device cuda
python -m whatif.pilot_train --phase adapt --device cuda
python -m whatif.pilot_eval --device cuda
python -m whatif.pilot_first_value --device cuda
# 三批原生评测；使用已发布分支，移除 --replay 即重新运行模拟器
for mode in original fixed reach; do
  python -m whatif.native_pilot --sim-python .venv/bin/python --mode "$mode" --device cuda --replay
done
# 只用已发布的首推价值模型复现确认集
python -m whatif.pilot_first_value --device cuda --replay
```

## 组合接触、换指迁移与两推决策

世界模型共享手指–物体、物体–物体的交互函数，复用同一组权重沿不同长度的接触路径推断。在同一物理粒子上计算两个动作对目标物块的位姿效果之差。观察事实动作的后继后更新物理信念，再按目标距离的预测分布在线选动作。

| 实验 | 已交付的分阶段中文报告 |
|---|---|
| 论文依据、实验划分与原生 Pushing/Picking 的投稿路线 | [实验方案](reports/generalization/00-plan.md) |
| 原生 TriFinger 的多物体分叉短推、可切换的接触路径 | [A · 场景与数据](reports/generalization/01-scenes.md) |
| 动作条件接触图、单一物理后验、显式反事实差值 | [B · 实现公式与效果预测](reports/generalization/02-effects.md) |
| 预测动作造成的新接触、断开接触和二维效果分布 | [B2 · 接触变化与分布质量](reports/generalization/06-edges-distribution.md) |
| 观察后选择第二推，与同结构稠密模型比较 | [C · 两推闭环](reports/generalization/03-decisions.md) |
| 训练用一根手指和至多两块，冻结后换指推三块 | [C2 · 三个动作方向](reports/generalization/04-action-frames.md) |
| 在新布局上连第一推也由目标驱动的控制器选择 | [C3 · 自适应第一推](reports/generalization/05-adaptive-control.md) |

仅用单块、双块场景训练，三块测试的末块效果误差由均匀先验的 9.50 mm 降到一次事实更新后的 4.86 mm。另一组 192 个新三块布局上，图模型按目标选择第一推后，两推终点距离从 17.81 降到 16.37 mm。同预算稠密关系模型对应为 17.12 和 15.44 mm。全部结果使用相同测试场景配对，分别报告 3 个训练种子。

[可直接使用的分叉数据](data/chains) · [检查点与逐场景结果](results/chains) · [代码](whatif)

### 复现多物体实验

采集环境沿用下文固定的 CausalWorld 版本。在独立的 PyTorch 环境拟合和评测。将命令中的 `cuda` 改为 `cpu` 可运行相同模型。

```bash
# 仿真环境；已有仓库中的原始数据时可直接运行后面的命令
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m whatif.chain_data --workers 12
python -m whatif.chain_scenes
# PyTorch 环境；约 19–23 秒/检查点（L40S），共 6 份
python -m whatif.chain_train --device cuda
python -m whatif.chain_eval --device cuda
python -m whatif.chain_eval --device cuda --splits frame0 frameplus frameminus --prefix frames_
python -m whatif.adaptive_control --device cuda
python -m whatif.chain_edges --device cuda
python -m whatif.chain_distribution --device cuda
```

数据与模型输出都带场景、动作或种子标识。多物体仓库中的 44,880 个不同快照–动作分支包括主训练/测试的 25,872 个、三个手指方向的 8,448 个和自适应首推组的 10,560 个。最后一组的四个数据文件复用了同一起始快照。

## 复现早期单块实验

服务器的采集环境为 Python 3.10.20，模型训练环境为 Python 3.13.5、NumPy 2.1.3、PyTorch 2.13.0。仿真和拟合可以使用独立解释器。早期单块模型在 CPU 上训练。已附原始分支数据，可直接从第二步开始。

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
