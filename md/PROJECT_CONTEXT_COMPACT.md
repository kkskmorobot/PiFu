# PiFuV2 精简上下文（2026-07-19）

## 项目与环境

- 项目：`E:\App\DaiMa\JetBrains\Project\PiFuV2`
- 课题：基于注意力机制改进 YOLO 的皮肤病识别算法研究
- GPU：RTX 5080 16 GB
- 解释器：`.venv\Scripts\python.exe`，Python 3.13.7
- PyTorch：2.8.0+cu129；Ultralytics：8.4.101；PyYAML：6.0.2
- 当前数据：`Data\processed_skin7_v3\data.yaml`
- 类别顺序：0 akiec、1 bcc、2 bkl、3 df、4 mel、5 nv、6 vasc
- train/val/test：8512 / 1504 / 1503 张；测试集在选定最终模型前不使用。

## 已确认的数据与模型问题

- runs3 的主要瓶颈是疾病分类，不是定位：验证集约 99% 能检测到病灶，正确类别样本的平均 Mask IoU 约 0.776–0.926。
- 主要错误：nv→mel 66、nv→bkl 34、mel→nv 26、bkl→mel 17、bkl→nv 12。
- 少数类的独立 lesion 很少：df 训练约 51、vasc 68、akiec 160；大量离线增强不等于新增临床病例。
- runs3 的 Mosaic/较强 HSV/缩放叠加离线增强，会破坏单个居中病灶的颜色、纹理和边缘特征。
- HAM10000 存在类别与采集设备、诊断确认方式相关的域偏差。
- 验证集独立 lesion 很少（df 约11、vasc约15、akiec约34），少数类指标需要多随机种子或 lesion 级交叉验证。

## runs3 基准结果（YOLO11s 实例分割）

- Box：P 0.7117、R 0.6784、mAP50 0.7444、mAP50-95 0.5957
- Mask：P 0.7112、R 0.6755、mAP50 0.7421、mAP50-95 0.5752
- Mask P/R：akiec .585/.592；bcc .694/.610；bkl .736/.591；df .617/.627；mel .566/.749；nv .939/.877；vasc .841/.682。
- 诊断文件：`C:\Users\ROBOT\Documents\皮肤识别\runs3_review\image_level_diagnostics.csv`

## runs4 当前实验（正在训练）

- 模型：官方 `yolo26s-sem.pt` 语义分割；约650万参数，YOLO26原生含C2PSA注意力。
- 语义标签：现有多边形自动转换为0–6疾病像素，并追加7=background。
- 输出：`runs4\semantic\skin7_yolo26s_sem_balanced`
- 配置：150 epochs、patience 30、batch 16、imgsz 640、AdamW lr0 .001、cls_pw .5。
- 增强：mosaic/mixup/cutmix/copy-paste=0，scale=.10，degrees=5，hsv_s/v=.10。
- 正式类别权重约：[1.04, 1.04, .92, 1.401, .933, .76, 1.51, .332]；重点提高df/vasc，降低nv/background。
- 640/batch16 峰值显存约5.67 GB。
- 截至第47轮：最佳为第46轮，mIoU .5819、pixel accuracy .9186；第47轮mIoU .536，训练仍正常。
- 当前存在训练/验证损失差距，验证mIoU在.54–.58间波动；不能仅凭当前结果断言过拟合已解决。
- 语义mIoU不能直接与runs3的实例Mask mAP比较。训练结束后运行 `evaluate_yolo26_sem_image_level.py`，计算图像级7类Precision/Recall/F1。

## 存档与恢复

- `weights\last.pt`：每个完整epoch更新，可恢复轮次、优化器、学习率调度器和早停状态。
- `weights\best.pt`：保存验证mIoU最佳模型。
- `save_period: 10`：已生成epoch0/10/20/30/40等独立快照。
- 断电时最多丢失尚未完成的当前epoch；重启后在PyCharm右键运行 `resume_yolo26_sem_v4.py`。
- 等价命令：`.venv\Scripts\python.exe train_skin7_yolo26_sem.py --resume runs4\semantic\skin7_yolo26s_sem_balanced\weights\last.pt`

## 后续实验顺序

1. runs4完成后只验证val，执行语义逐类mIoU与图像级Macro-P/R/F1分析。
2. 若runs4确有提升，runs5保持数据、损失、增强和随机种子完全不变，只加入CBAM，形成注意力消融。
3. 后续候选：病灶引导的多尺度通道+空间注意力、辅助全局疾病分类头。
4. 最终模型确定后才评估test；论文报告多随机种子/lesion级置信区间、Macro-F1和逐类召回率。
