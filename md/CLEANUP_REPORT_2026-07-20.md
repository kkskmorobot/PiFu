# PiFuV2 备份、整理与清理报告

完成时间：2026-07-20

## 1. 备份结果

备份位置：`E:\App\DaiMa\JetBrains\Project\BeiFen\PiFuV2`

- 排除：`Data` 和所有 `.venv*` 目录
- 已备份文件：268
- 已备份字节：2,310,555,920
- 逐文件 SHA-256 不一致：0
- 缺失：0
- 校验清单：`BACKUP_MANIFEST_SHA256.csv`
- 备份说明：`README_BACKUP.md`

## 2. 最终项目结构

```text
PiFuV2/
├─ .idea/                         PyCharm 配置
├─ .venv/                         唯一保留的有效环境
├─ Data/
│  └─ processed_skin7_v3/         唯一保留的数据集
├─ docs/                          项目交接与清理报告
├─ experiments/
│  ├─ 01_YOLO11s_V1_Baseline/
│  ├─ 02_YOLO11s_V2_DataBalance/
│  ├─ 03_YOLO11s_V3_NV3200/
│  └─ 04_YOLO26s_Semantic_C2PSA/
├─ models/                        yolo11s-seg.pt、yolo26s-sem.pt
├─ tools/                         数据、训练、恢复、评价脚本
└─ PROJECT_INDEX.md               项目入口索引
```

未来正式实验从 `experiments\05_...` 开始编号。

## 3. 数据清理

保留：

- `Data\processed_skin7_v3`
- train/val/test：8512 / 1504 / 1503
- V3 包含 images、labels、metadata、data.yaml 和 summary.json，可直接训练。

已永久删除：

- `Data\train`：原始 HAM10000 工作副本
- `Data\test`：原始官方测试下载工作副本
- `Data\processed_skin7`：V1 处理数据
- `Data\processed_skin7_v2`：V2 处理数据

上述旧数据不在本次非数据备份中。如需重建，只能使用用户另行保存的官方原始备份。

## 4. 虚拟环境说明

原来出现多个 `.venv*` 是因为解释器修复和 YOLO26 安装过程中留下了历史尝试：

- `.venv`：当前有效环境，Python 3.13.7、PyTorch 2.8.0+cu129、Ultralytics 8.4.101，保留。
- `.venv_broken_python313`：旧的损坏环境，已删除。
- `.venv_python312_empty`：未完成的空 Python 3.12 环境，已删除。
- `.venv_yolo26`：YOLO26 环境尝试残留，已删除。

当前 `.venv` 约 7.965 GiB，主要体积来自 PyTorch、CUDA 运行库和训练依赖，这是有效环境的正常占用，不是重复环境。

## 5. 其他已删除内容

- `_smoke`：模型烟雾测试产物
- `.downloads`：安装下载缓存
- `.piptemp`：空 pip 临时目录
- 根目录与 tools 中的 `__pycache__`
- 整理后遗留的旧 `runs`、`runs2`、`runs3` 目录
- `yolo11n.pt`、`yolo26n.pt`：仅用于早期试跑的小模型
- 三个无效虚拟环境

正式 runs1~runs4 没有删除，已移动到编号 experiments 目录。

## 6. 空间变化

- 本次永久删除：11,262,371,931 字节，约 10.489 GiB
- 非数据备份新增：约 2.152 GiB
- E 盘净释放约 8.337 GiB
- 清理后项目内容（不计算 `.venv` junction）：约 4.375 GiB
- 加上有效 `.venv` 后项目实际占用约 12.34 GiB

清理后的主要占用：

- `.venv`：约 7.965 GiB
- V3 数据：约 2.427 GiB
- 四次实验：约 1.917 GiB
- 预训练模型：约 0.032 GiB

## 7. 完整性验证

四次正式实验的 `best.pt` 与整理前备份逐一计算 SHA-256，结果全部一致：

- 第一次：一致
- 第二次：一致
- 第三次：一致
- 第四次：一致

runs4 仍保留：

- `best.pt`
- `last.pt`
- `epoch0.pt`
- `epoch10.pt`
- `epoch20.pt`
- `epoch30.pt`
- `epoch40.pt`
- `epoch50.pt`
- `epoch60.pt`
- `epoch70.pt`
- `epoch80.pt`
- `epoch90.pt`

## 8. 可运行性验证

整理后的公共脚本已适配 `tools` 目录：

- Python 语法编译通过
- 当前 `.venv` Python 版本：3.13.7
- YOLO11 V3 dry-run 通过
- YOLO26 semantic V3 dry-run 通过
- V3 三个 split 的图片/标签与类别统计通过
- dry-run 未启动训练，也未运行 test 推理
- 将来重跑会写入各实验自己的 `reruns`，不会重新在项目根目录生成 runs/runs2/runs3/runs4

## 9. 使用入口

先阅读：`PROJECT_INDEX.md`

详细历史：`docs\PROJECT_HANDOFF_2026-07-20.md`

当前第四次实验：`experiments\04_YOLO26s_Semantic_C2PSA`

下一次实验应建立：`experiments\05_...`
