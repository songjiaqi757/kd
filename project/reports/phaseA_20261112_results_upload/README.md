# Phase A 完整结果包

本目录完整保存服务器生成的 `phaseA_20261112_results.tar.gz`，按原字节分为 3 卷，每卷不超过 40 MiB；没有重压缩或删减归档内容。归档包含完整逐样本四象限表、报告、汇总统计、协议、源码与运行记录。

逐卷及完整包的大小、SHA256 见 `manifest.json`。

在本目录执行：

```bash
sha256sum -c SHA256SUMS
cat phaseA_20261112_results.tar.gz.part* > phaseA_20261112_results.tar.gz
sha256sum -c ARCHIVE_SHA256SUM
tar -xzf phaseA_20261112_results.tar.gz
```

macOS 可将 `sha256sum` 替换为 `shasum -a 256`。解压后得到 `icd_review_revision_202611/`；若要将详细表恢复到仓库对应目录，可把解压目标设为 `project/reports/`。

[可浏览的主报告](../icd_review_revision_202611/phaseA_final_report.md) · [Phase B 决策](../icd_review_revision_202611/phaseB_decision.md)
