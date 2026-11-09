# 新旧 MOSI Baseline 协议差异审计

审计对象：uniform_main_v1/mosi 与 icd_improvement_mosi_20261107。全部读取和产物仅位于 Qi /data/sjq/kd；未启动 GPU，未读取 Official-test 样本或指标。

## 结论

0.7312/0.7601 与 0.7553/0.7895 来自不同源码和训练预算，不能仅据最终 MAE 推断模型退化或将差值归因于单一实现变化。教师及数据资产在历史路径 /ai/sjq/kd → /data/sjq/kd 归一化后相同；旧新均存在 Full KD 单 Probe、ICD 交互三 Probe 的监督来源不匹配，第二阶段必须统一三 Probe。

旧 run_config 标注 checkpoint_selection=test_mae，但与其记录 SHA 完全匹配的 Git b50cd6c 训练脚本，第502–507及538–539行仍以 validation MAE 写 best.pt 并评估生成 report；报告中的两个旧 MAE 也恰为 history 中的 validation 最低值。test_mae 影响的是训练不早停，并另行生成历史 test_sweep。应将这些引用值称为历史 validation-selected report 值，同时说明其上层历史流程曾使用 test sweep；不能把 config 标记直接等同于这两个数来自 test-selected checkpoint。

## 逐项核对

| 项目 | 旧 uniform_main_v1 | 第一轮 improvement | 第二阶段处理 |
|---|---|---|---|
| 1 Teacher Probe/checkpoint | Full KD TAV=Probe2026；ICD base TAV=2026，交互=2026/2027/2028 | 相同；三个 teacher_probe.pt 当前哈希见下表 | 全部组 Full KD及ICD统一相同三 Probe；清单记录实际checkpoint哈希 |
| 2 教师预测/交互目标 | frozen_assets含单Probe预测、ensemble_targets、interaction_targets、per-probe empty baseline | 路径归一化后完整frozen_assets相同；prediction哈希匹配；Probe2026 report记录哈希已陈旧 | 重新冻结实际文件SHA；保持 Möbius与empty定义 |
| 3 学生初始化/可训练参数 | Qwen3-0.6B+WavLM+VideoMAE，subset transformer fusion；819309176总/42650120可训练；seed13 | 参数数目、build_model逻辑相同；初始参数哈希未存，无法证明逐位相同 | 对8组记录并核对初始化参数哈希；依赖源码一并冻结 |
| 4 LoRA | video_layers12/rank8/alpha16/dropout0.05，video seed=100013，checkpointing开启 | 相同；T/A/V LoRA范围由共同M3实现决定 | 相同并记录各参数组统计 |
| 5 数据窗口/聚合 | 1286训练窗口、1284 utterances；229验证窗口/utterances；max30_overlap5；utterance按aggregation_weight加权 | 同一manifest内容（路径迁移改变字节SHA），验证10个source videos，训练52个，无交集 | 保持划分及顺序；不得将1286窗口误作1284 utterances |
| 6 Loss/reduction/样本权重 | Task SmoothL1+0.5 CE；KD SmoothL1+KL；T=2；teacher Tcal2026=1.1912176609039307；ICD λ1；weighted mean | 相同数学目标，参数显式化；ICD由先坐标mean再weightedmean改为先weightedmean各坐标再坐标mean；浮点求和顺序变化 | 明确记录reduction；c0保留KD KL；三Probe概率须先各自softmax再算术平均 |
| 7 Optimizer/LR/参数分组 | AdamW trainable list单组，LR1e-4/wd.01/clip1 | baseline仍同一单组；scheduler constant；新增分组与cosine选项未用于E0/control | Phase1固定单组恒定LR；只有后续scheduler对照改变调度 |
| 8 DataLoader/随机种子 | seed_everything(13,True)，Generator.seed13，train shuffle=True，validFalse，batch8/workers2 | 相同声明；没有历史初始化哈希/完整运行库版本可佐证逐位复现 | 冻结初始化哈希、采样协议及环境；诊断不得扰动训练随机数 |
| 9 早停/checkpoint | max20/min8/patience7；test_mae分支禁用早停，全20轮；report由valid minimum选择：Full19，ICD15 | max30/min1/patience7，validMAE/tie1e-5；Full29轮/best22，ICD11轮/best4；每轮保存 | max40/patience8/validMAE；若任组best>=36，全部统一50预算 |
| 10 源码版本/SHA | train_main_table_kd SHA81ba82...；main_table_kd SHA0bed4d...；可从Git精确找回 | 独立snapshot trainer SHAa9c50d...；main_table_kd SHA1eec06...；历史依赖新增shared-dropout/explicit/GP等分支 | 独立train_icd_confirmatory.py；冻结完整依赖与SHA，不修改历史入口 |

## 引用 MAE 及预算证据

| Run | Validation MAE | 最优 epoch | 实际 epoch | 声明 selection |
|---|---:|---:|---:|---|
| old_full | 0.731166541 | 19 | 20 | test_mae |
| old_icd | 0.760095126 | 15 | 20 | test_mae |
| first_full | 0.755313035 | 22 | 29 | valid_mae |
| first_icd | 0.789549493 | 4 | 11 | valid_mae |
| historical_gp | 0.705557975 | 15 | 20 | valid_mae |

## 教师身份与实际 SHA256

三个教师 checkpoint均为 teacher_probe.pt。预测资产哈希与历史冻结值一致；report是元数据资产，当前2026报告SHA与历史嵌入值不同，不等价于教师权重或预测发生变化。

| Seed | 最优 epoch | Checkpoint SHA256 | Prediction SHA256 |
|---|---:|---|---|
| 2026 | 12 | ff525dfaa189a21c55f1f5254d85d4c86ac64d2ed84d8863ba311fd552bb0408 | 860c87e36aeaaa0578d9477df13cb934106c11d88a37a6f6b2727bd133183ffb |
| 2027 | 10 | 67b629ee3f6589a53c3bd3bf825a7a5cb7df56f7525f904dd4eb8c34f572d0a2 | dee515299207effd6dde2f45d5ea006a9fc0927466d383650ffd1b61f54f79fb |
| 2028 | 12 | b6c36640c689f57d8b45dd9a79d575c91bc73fee0ea7b2b99ee2c4dfb411e8ed | c06831d7a2e1c490837bf2a8a6818564119efb8b19ee35bdb1cda1bb2c4fdcb8 |

当前检测到的历史哈希不一致（所有这些差异需在新manifest以实际SHA冻结）：

- /data/sjq/kd/outputs/experiments/main_table_v1/mosi/base_assets/protocol.json: recorded a2403723386b781a6b5b80c036f0d47cc3f9cf83efdf27dfa441cbbd870ae8be; actual b9622a1e01205d031ddbfa5ef4d3ae1e53c00e12309ea6e24ee0cb9d0b3d2970.
- /data/sjq/kd/dataset/cmu_mosi/manifests/official_train_valid_windowed.jsonl: recorded 0f64f372da77e42685eb0e490fdb653a1d05ae1d3dfbddcd3bc9e56e3b360c7c; actual ff94c37bcab1729a2e948c3f161c90fd92168229e203144f70de00e0440d7d3e.
- /data/sjq/kd/outputs/experiments/main_table_v1/mosi/probes/seed2026/report.json: recorded d2d5104980364ce54dee6c01d8e6d941be223a1ee37a92008fc1de5517ccd99e; actual 6e75843b29fb29a6ec1ad26e4f7b30e697d009d62c984f4bdbdccffc83b10cc6.

Empty baseline仍定义为每Probe官方train TAV utterance mean：[0.2844906165580065,0.19138739822610829,0.19990882057206258]；student baseline=0.22526227845205912。

## GP 公平复现实务

现有 project/scripts/train_main_table_kd.py:423 的 projected_auxiliary_backward 对所有可训练参数，将 base=task+FullKD 与 aux=λ*七坐标mean 分别求梯度。若全局 dot<0，将 aux 减去 dot/||base||² * base，再与 base 相加；最后共用clip1与AdamW。其普通ICD前向顺序TAV先、其余子集随后；GP不会切换shared dropout或explicit fusion。复现时必须让base使用本轮c及三Probe目标，aux使用本轮λ，普通与GP共用同一loss_terms；不可直接调用旧 uniform_interaction_components 保留单Probe/固定CE目标。

历史GP seed13=0.705558、best15/max20/min20/patience20、validMAE；不能直接与新配置排名。三种子历史结果仅作背景，本次开发仍seed13。

## 证据位置与边界

baseline_audit.json 保存5个run的原始配置、report、history minimum、证据SHA、源代码哈希、数据计数及当前教师哈希；protocol_source_diff.patch 保存从Git b50cd6c到第一轮snapshot的相关源码差异。旧trainer和main_table_kd模块的SHA与run_config逐一匹配，旧未记录的transitive依赖使用该Git版本仅作上下文证据。

未发现历史初始化参数哈希，因此不能声称旧新初始化逐位相同；历史每run运行库版本不完整。相关函数的表面等价不足以因果解释最终MAE差异。数学等价的ICD求和顺序在浮点下可能改变轨迹，但本审计没有量化该因素。

源码与运行证据：

- old_full: /data/sjq/kd/outputs/experiments/uniform_main_v1/mosi/students/full_kd_seed13/run_config.json; report.json; history.json.
- old_icd: /data/sjq/kd/outputs/experiments/uniform_main_v1/mosi/students/uniform_interaction_seed13/run_config.json; report.json; history.json.
- first_full: /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/runs/control_full_kd_seed13/run_config.json; report.json; history.json.
- first_icd: /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/runs/E0_uniform_seed13/run_config.json; report.json; history.json.
- historical_gp: /data/sjq/kd/outputs/experiments/mosi_research_v1/uniform_interaction_gradient_projected_seed13/run_config.json; report.json; history.json.
- project/scripts/train_main_table_kd.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 81ba82e09887050b95702a2c5d63a4648c819dc039c599600d703847ce796ba6; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/scripts/train_icd_improvement.py SHA256 a9c50d995cc6d8097f43184789863a253edc040e3878f356c802198a0604baf7.
- project/src/rdid_mosei/main_table_kd.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 0bed4d346e65ed3bf423f76735423bf25a80c031ab6e5e30b30efafcac298610; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/src/rdid_mosei/main_table_kd.py SHA256 1eec0626e0069a7dc9c4d350591c7620c65bcb32f743126000224bf2a92a1e85.
- project/src/rdid_mosei/student.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 407152e40e89fb10e82be0922c3a6364e0a914fdd563c3d344e15f8923798b72; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/src/rdid_mosei/student.py SHA256 43aa48f30abc8855b38556ffd01310eaa2d159756c7d3c03ff63878ee1a6a0dc.
- project/src/rdid_mosei/tav_distillation.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 2fa00af5d242b5e6db7eaed6c5d7c3ec19ac8294b3cbdb48369185dcfca92695; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/src/rdid_mosei/tav_distillation.py SHA256 3b8b5838ad16dc53d52a9a2167c1d449fb505469822e58bd8bee64312dd5cb23.
- project/scripts/train_student_baseline.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 b067131f614cd4fa8ddff414ad15e76c77c47ca4dae597ca039a4f0b9df83147; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/scripts/train_student_baseline.py SHA256 26522066e523cc3f907f5a4bb20f1397a22cb95ee65d6db54326a25c42826f93.
- project/scripts/train_tav_distillation.py: Git b50cd6c8abe8c9c311edd84cec6724ceaf6fe699 SHA256 3a7c2f31691d59f66e5573e8da73ce9fdae62a9baa764917ce491d334687ed48; first-round /data/sjq/kd/outputs/experiments/icd_improvement_mosi_20261107/source/project/scripts/train_tav_distillation.py SHA256 3a7c2f31691d59f66e5573e8da73ce9fdae62a9baa764917ce491d334687ed48.

## 路径迁移哈希的精确复核

三处历史SHA不一致均已确定为纯路径变更：对当前文件字节进行 /data/sjq/kd → /home/wy/sjq/kd 的反向替换后，SHA256逐一恰与冻结协议记录相同。这证明manifest、base protocol和Probe2026 report相对于其原记录内容仅改变了路径前缀，没有改变数据标签、数值或报告内容。baseline_audit.json中的reverse_path_migration_sha256与explained_by_path_only_migration保留复核结果。新协议仍必须冻结当前实际SHA，而不能沿用旧字节哈希。
