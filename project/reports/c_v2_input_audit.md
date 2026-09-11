# C-v2 输入与缓存审计

状态：**pass_with_v2_gaps**

- Manifest windows：18,278
- Student cache：18,278 index rows / 18,278 feature files
- Teacher cache：127,946 rows，feature shape=[127946, 2048]
- Manifest SHA256：`21779839489bb1c9e14baa2e287399e4fadfefe95db7499efa45dbadd97a9380`
- Legacy cache 可安全读取：True
- C-v2 前需升级索引或重建：True

## Errors

- None

## V2 gaps

- student index lacks v2 identity/fingerprint fields: feature_sha256, manifest_sha256, video_id, window_end, window_index, window_start
- teacher index lacks v2 identity/fingerprint fields: feature_sha256, manifest_sha256, window_end, window_start
- model paths are recorded, but immutable model revision/file hashes are not recorded in the legacy cache config

Official test 未读取、未评估。
