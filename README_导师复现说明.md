# 论文项目复现

本项目包含当前论文使用的 16 组模型、完整训练与测试数据的索引、原训练器与评估代码、精简后的精确评测权重、历史结果及最新论文。项目和数据是两个独立目录，放在同一父目录即可移动使用：

```text
project/
  PiFu/
    reproduction/          安装、训练、验证、测试、校验与前端入口
    tools/                 原模型定义、参数和历史实验记录
    models/                重新训练所需预训练权重
    weights/               16个选定检查点的无损FP32 EMA
    experiments/           原始曲线、CSV、JSON和图片，不含大量中间权重
    paper/                 当前论文Word/PDF及图表证据
    vendor/                当前使用的Ultralytics源码、许可证及绘图字体
    outputs/               每次复现运行的新结果（运行后自动生成）
  PiFuData/
    processed_imaplusplus_binary_v1/
    processed_isic2019_classification_incremental_v1/
    processed_isic2018_task1_incremental_v1/
    processed_skin7_v3/     仅界面演示用的140张均衡子集
```

## 先安装环境

原机器使用 Windows、Python 3.13.7、PyTorch 2.8.0+cu129、Ultralytics 8.4.101 和 RTX 5080。原虚拟环境是指向另一项目的链接，未复制。首次安装依赖需要联网；框架源码及模型均已随包保留。

在 PiFu 目录的 PowerShell 7 中执行：

```powershell
pwsh -ExecutionPolicy Bypass -File reproduction/install.ps1
```

没有 NVIDIA GPU 时可以用 `-Device cpu` 安装 CPU 环境进行功能检查。CPU 速度较慢，原论文完整训练和数值测试采用 CUDA。若已有其他 Python 安装，使用 `-PythonExe '完整路径/python.exe'` 指定。

下面命令都在 PiFu 目录运行。若 IDE 中直接运行文件，请选择本项目 `.venv` 解释器。

## 直接执行与修改参数

复现副本不要求原作者授权、READY 标记或一次性测试许可；完成后允许重复训练和测试。每次运行自动创建带时间戳的新输出目录。训练 YAML 可修改，默认值保留论文原设置。

```powershell
# 先做一次小样本、一轮训练的功能检查（不是论文结果）
.venv/Scripts/python.exe reproduction/run.py train --run lgx --smoke

# 完整重训LGX，默认100轮上限及原始早停、种子、数据划分
.venv/Scripts/python.exe tools/lgx/train_lgx_from_pycharm.py

# 按导师需要修改轮数、batch、设备等；也可直接编辑该实验原YAML
.venv/Scripts/python.exe reproduction/run.py train --run lgx --epochs 10 --batch 8 --workers 0 --device 0

# 按需要调整早停；patience=0关闭早停
.venv/Scripts/python.exe reproduction/run.py train --run lgx --patience 0 --early-stop-start-epoch 0

# 仅生成配置和运行计划，不启动训练
.venv/Scripts/python.exe reproduction/run.py train --run lgx --prepare
```

原有16个 `tools/<run>/train_<run>_from_pycharm.py` 训练入口均已接到可迁移流程，可以直接点击运行。模型前向计算、损失公式和论文默认参数保持原样。原先与本机历史预检绑定、与训练无关的授权/PAD数据检查不再作为运行门槛；可调参数也不再要求只能等于论文固定值。文件存在、模型形状及必要数据格式错误仍会正常报错。

`md` 和 `experiments` 内的文档保留历史实验记录。其中提到的原作者审批、冻结期或禁止重复测试，不是本复现副本的执行要求。新的使用说明以本文件为准。

|实验类别|可用run ID|种子|
|---|---|---|
|Plain基线|run18 / run22 / run24|42 / 43 / 44|
|ECA|run21 / run23 / run25|42 / 43 / 44|
|ECA + Spatial7|run27 / run28 / run26|42 / 43 / 44|
|CBAM|run33|43|
|初版LGLA（失败实验）|lg1|43|
|最终LGLA|lg2 / lgx / lg3|42 / 43 / 44|
|结构对照|ab1 / ab2|43 / 43|

完整数据：联合训练23,462张、验证1,744张；内部测试634张，外部分割测试89张。划分、标签、图像与掩膜保持原始字节。训练不使用测试/外部数据。原训练增强和早停均属于算法设置，并未为打包而改变。

## 重跑现有选定权重的验证与测试

```powershell
# 对论文选定的LGX权重重算开发集指标
.venv/Scripts/python.exe reproduction/run.py validation --run lgx

# LGLA三个种子的正式固定测试（可重复运行）
.venv/Scripts/python.exe tools/test2/test.py

# Plain三个种子的正式固定测试（可重复运行）
.venv/Scripts/python.exe tools/test1/test.py

# 仅测试单个种子
.venv/Scripts/python.exe reproduction/run.py evaluate --suite test2 --run lgx

# 评价导师自己新训练的检查点（目录使用实际输出路径）
.venv/Scripts/python.exe reproduction/run.py validation --run lgx --bundle outputs/实际训练目录/training/checkpoint_bundles/aliases/best_joint_score.json
.venv/Scripts/python.exe reproduction/run.py evaluate --suite test2 --run lgx --bundle outputs/实际训练目录/training/checkpoint_bundles/aliases/best_joint_score.json
```

加 `--limit 2` 可快速跑少量样本，仅用于检查流程，输出明确标记为非正式结果。正式评价使用完整数据、分割阈值0.5、分类argmax及原分组/汇总规则。内部测试仅56张有MEL/NV分类标签，不能据此报告完整八类测试Macro-F1。

训练输出中的 `checkpoint_bundles/aliases/best_joint_score.json` 记录新一轮训练按原验证规则选中的epoch。`--bundle` 可指定这个别名文件，也可指定某个epoch目录；新的训练评价会单独标记，不会被写成原始冻结模型的结果。若要评价普通部署权重，可以用 `--weights 路径/best.pt` 替代 `--bundle`；普通权重可能经过FP16舍入，不等同于无损FP32 EMA。

本包 `weights/<run>/frozen_eval.pt` 以原选定 bundle 的结构为模板，严格加载 `runtime_state.pt::training_state.ema` 的完整FP32状态；导出再加载后的所有状态张量逐元素相同。没有重新量化、剪枝或使用可能选中另一epoch的 `best.pt` 代替。优化器、梯度缩放器、随机数状态及中间epoch未打包。来源路径、原始文件SHA-256、导出SHA-256和一致性结果见 `reproduction/frozen_models.json`。这些权重可直接评价；重新训练从 `models` 中的原预训练权重开始。

## 前端

```powershell
.venv/Scripts/python.exe QianDuan/app.py
```

浏览器打开 `http://127.0.0.1:8765`。界面可选16个论文模型；默认LG3，与论文界面示例一致。前端随机/批量演示使用140张均衡旧skin7样本，不用于正式训练或论文测试。界面字体/样式仍引用原前端CDN，浏览器首次加载需要网络。

## 数据位置与完整性

数据目录默认是项目相邻的 `PiFuData`。如果另存其他路径，可以设置：

```powershell
$env:PIFU_DATA_ROOT = '其他磁盘:/你的路径/PiFuData'
```

原始证据文件中的 E 盘路径保留用于追溯；复现入口会生成当前机器上的运行配置，不会据此读取原 E 盘项目。

```powershell
# 传输后检查文件；不需要安装机器学习依赖
python reproduction/verify.py --full

# 检查原始实验配置与数据是否仍符合论文归档记录
.venv/Scripts/python.exe reproduction/run.py check
```

完整性检查用于发现传输损坏。修改配置后，历史一致性审计可能提示不同，但不禁止导师运行自己的训练配置。当前环境安装清单与框架审计在 `reproduction/environment.json`。

## 简化范围

包内足够重训和评价当前论文上述16组实验。其他早期探索的结果图片/CSV仍保留用于追溯，但不提供早期Run1–9的完整训练数据与全部历史权重。已移除原始数据解压副本、未进入论文的PAD数据、重复预检文件、冗余训练状态、旧虚拟环境、旧论文渲染缓存。数据从已冻结的处理后文件开始；重建这些处理文件的原始下载包不在本包内。

请把 **PiFu 与 PiFuData 两个目录一起发给导师**。运行后生成的 `.venv`、`outputs`、`runtime` 无需再次发送。实际迁移验收范围和结果请见根目录 `VERIFICATION.json`。
