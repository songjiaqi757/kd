# KD / RDID-MSA Workspace

本目录是 RDID-MSA 实验的唯一工作区。代码、文档和小型统计由 Git 管理；数据、模型和实验产物仅保存在本地。

## 快速入口

- [文档导航与版本地位](docs/README.md)
- [最新冻结的 TAV/P0 结果](docs/TAV主实验结果汇总_20260917.md)
- [主表对比与消融方案](docs/RDID-MSA_主表对比与消融实验方案_20260917.md)
- [实验实现说明](docs/RDID-MSA_实验实现说明_20260917.md)
- [代码与复现命令](project/README.md)

运行中任务以 `outputs/experiments/*/status.json` 为准，文档只记录已冻结结果。Official test 尚未使用。

## 目录边界

- `project/`：源码、配置、测试、可重算脚本和机器可读报告。
- `docs/`：研究方案、人类可读结果、审计记录和历史快照。
- `dataset/`：整理后的 MOSEI/MOSI 与只用于溯源的 source 数据。
- `model/`：教师模型和三个学生编码器。
- `outputs/`：教师缓存、特征、训练检查点、预测、日志和运行状态。
- `external/`：外部方法的来源、适配审计、补丁和本地上游仓库。
- `environment/`：Conda、pip、GPU 和 ffmpeg 复现快照。

Conda 环境位于 `/home/wy/sjq/miniconda3/envs/kd`。验证命令：

```bash
/home/wy/sjq/miniconda3/bin/conda run -n kd pytest -q
```

## 产物保留规则

- 原始数据、清洗后数据、模型权重、特征缓存和最佳 checkpoint 不自动删除。
- 已完成实验保留 `best.pt`、配置、历史、预测和报告；`last.pt` 只用于未完成任务续训。
- smoke/preflight 完成后保留配置、指标和审计记录，不长期保留其大型 checkpoint。
- 中断运行仅在已有同配置的完整替代结果时删除。

`dataset/`、`model/`、`outputs/` 和 `external/original/` 已由 `.gitignore` 排除，不上传 GitHub。
