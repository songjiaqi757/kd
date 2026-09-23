# Efficiency / Deployment figure caption

**English.** Efficiency and deployment analysis on CMU-MOSI. (a) Test MAE
versus total training wall-clock time for five representative methods. The
dotted line connects the non-dominated methods when minimizing both quantities.
(b) Measured resource change of Uniform Interaction relative to Full KD. Uniform
reduces Test MAE by 3.15% while retaining the same number of trainable and
deployed parameters and the same inference memory footprint. Results use seed
13. Training time covers the epochs actually executed, including early stopping;
inference time is a single-run measurement. Test MAE is selected from the
post-training checkpoint sweep, and the test set was excluded from training.

**中文说明。** CMU-MOSI 上的效率和部署分析。(a) 五种代表性方法的 Test MAE 与
训练总墙钟时间；虚线连接同时最小化两项指标时的非支配方法。(b) Uniform Interaction
相对 Full KD 的实测资源变化。Uniform 将 Test MAE 降低 3.15%，同时保持相同的可训练
参数量、部署参数量和推理显存。结果使用 seed 13；训练时间统计实际执行轮次并包含
early stopping，推理时间为单次测量。Test MAE 来自训练结束后的 checkpoint sweep，
测试集没有参与训练。

## Reproduce

```bash
/home/wy/sjq/miniconda3/envs/kd/bin/python \
  project/scripts/plot_efficiency_deployment.py
```
