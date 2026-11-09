# MOSI official-test: first + second round

Manifest: `/data/sjq/kd/dataset/cmu_mosi/manifests/official_test_windowed.jsonl` (686 windows / 686 utterances)

| round | run | checkpoint | epoch | valid MAE | test MAE | test Pearson | test Acc-2 | test F1 | test Acc-7 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| round1 | E0_uniform_seed13 | valid_acc2_nonzero | 3 | 0.8179 | 0.8336 | 0.7575 | 0.8293 | 0.8299 | 0.4344 |
| round1 | E0_uniform_seed13 | valid_mae | 4 | 0.7895 | 0.7344 | 0.7892 | 0.8415 | 0.8421 | 0.4431 |
| round1 | E1_l1_seed13 | valid_acc2_nonzero | 16 | 0.8170 | 0.7323 | 0.7996 | 0.8521 | 0.8517 | 0.4388 |
| round1 | E1_l1_seed13 | valid_mae | 20 | 0.7909 | 0.7348 | 0.7986 | 0.8491 | 0.8498 | 0.4402 |
| round1 | E2_no_ce_seed13 | valid_acc2_nonzero | 11 | 0.7514 | 0.7534 | 0.8180 | 0.8613 | 0.8620 | 0.4169 |
| round1 | E2_no_ce_seed13 | valid_mae | 9 | 0.7192 | 0.7239 | 0.8065 | 0.8552 | 0.8559 | 0.4431 |
| round1 | E3_no_kl_seed13 | valid_acc2_nonzero | 12 | 0.8051 | 0.7478 | 0.7793 | 0.8399 | 0.8391 | 0.4388 |
| round1 | E3_no_kl_seed13 | valid_mae | 16 | 0.7856 | 0.7388 | 0.7764 | 0.8476 | 0.8479 | 0.4431 |
| round1 | E4_weight025_seed13 | valid_acc2_nonzero | 25 | 0.7632 | 0.7096 | 0.8029 | 0.8643 | 0.8645 | 0.4373 |
| round1 | E4_weight025_seed13 | valid_mae | 26 | 0.7334 | 0.7147 | 0.8112 | 0.8506 | 0.8512 | 0.4738 |
| round1 | E4_weight050_seed13 | valid_acc2_nonzero | 17 | 0.7761 | 0.7690 | 0.8023 | 0.8445 | 0.8455 | 0.4373 |
| round1 | E4_weight050_seed13 | valid_mae | 13 | 0.7339 | 0.6800 | 0.8165 | 0.8598 | 0.8599 | 0.4606 |
| round1 | E5_warmup_seed13 | valid_acc2_nonzero | 5 | 0.9278 | 0.8061 | 0.7841 | 0.8521 | 0.8517 | 0.3557 |
| round1 | E5_warmup_seed13 | valid_mae | 14 | 0.7606 | 0.7423 | 0.8055 | 0.8552 | 0.8560 | 0.4315 |
| round1 | E6_order_seed13 | valid_acc2_nonzero | 8 | 0.8667 | 0.7482 | 0.7894 | 0.8521 | 0.8512 | 0.4169 |
| round1 | E6_order_seed13 | valid_mae | 13 | 0.7745 | 0.7142 | 0.8051 | 0.8537 | 0.8533 | 0.4417 |
| round1 | E7_video_coordinates_seed13 | valid_acc2_nonzero | 9 | 0.8546 | 0.7927 | 0.7781 | 0.8399 | 0.8395 | 0.3819 |
| round1 | E7_video_coordinates_seed13 | valid_mae | 14 | 0.7729 | 0.7566 | 0.7945 | 0.8399 | 0.8409 | 0.4402 |
| round1 | E8a_grouped_lr_seed13 | valid_acc2_nonzero | 2 | 0.8597 | 0.8424 | 0.7372 | 0.8216 | 0.8199 | 0.3499 |
| round1 | E8a_grouped_lr_seed13 | valid_mae | 17 | 0.7567 | 0.7163 | 0.7935 | 0.8430 | 0.8430 | 0.4431 |
| round1 | E8b_cosine_seed13 | valid_acc2_nonzero | 17 | 0.7430 | 0.7129 | 0.7929 | 0.8338 | 0.8342 | 0.4592 |
| round1 | E8b_cosine_seed13 | valid_mae | 15 | 0.7368 | 0.7193 | 0.7930 | 0.8430 | 0.8436 | 0.4461 |
| round1 | control_full_kd_seed13 | valid_acc2_nonzero | 25 | 0.7996 | 0.7131 | 0.7836 | 0.8430 | 0.8419 | 0.4490 |
| round1 | control_full_kd_seed13 | valid_mae | 22 | 0.7553 | 0.7309 | 0.7914 | 0.8567 | 0.8573 | 0.4431 |
| round2 | F0_full_kd_ce | valid_acc2_nonzero | 5 | 0.8804 | 0.7560 | 0.7833 | 0.8506 | 0.8497 | 0.4155 |
| round2 | F0_full_kd_ce | valid_mae | 13 | 0.7453 | 0.7174 | 0.7914 | 0.8430 | 0.8434 | 0.4621 |
| round2 | F1_full_kd_no_ce | valid_acc2_nonzero | 17 | 0.7628 | 0.6779 | 0.8142 | 0.8598 | 0.8587 | 0.4563 |
| round2 | F1_full_kd_no_ce | valid_mae | 19 | 0.7548 | 0.7091 | 0.8152 | 0.8506 | 0.8514 | 0.4431 |
| round2 | F2_icd_ce_lambda1 | valid_acc2_nonzero | 2 | 0.9718 | 0.9096 | 0.7498 | 0.8186 | 0.8147 | 0.3149 |
| round2 | F2_icd_ce_lambda1 | valid_mae | 6 | 0.7768 | 0.7577 | 0.7999 | 0.8430 | 0.8439 | 0.4213 |
| round2 | F3_icd_no_ce_lambda1 | valid_acc2_nonzero | 5 | 0.8576 | 0.7536 | 0.7971 | 0.8399 | 0.8383 | 0.4082 |
| round2 | F3_icd_no_ce_lambda1 | valid_mae | 23 | 0.7697 | 0.7036 | 0.8139 | 0.8567 | 0.8574 | 0.4475 |
| round2 | F4_icd_ce_lambda05 | valid_acc2_nonzero | 20 | 0.7779 | 0.7041 | 0.7960 | 0.8506 | 0.8509 | 0.4592 |
| round2 | F4_icd_ce_lambda05 | valid_mae | 17 | 0.7583 | 0.7449 | 0.7882 | 0.8384 | 0.8394 | 0.4329 |
| round2 | F5_icd_no_ce_lambda05 | valid_acc2_nonzero | 10 | 0.7714 | 0.7353 | 0.7884 | 0.8308 | 0.8307 | 0.4592 |
| round2 | F5_icd_no_ce_lambda05 | valid_mae | 15 | 0.7643 | 0.7347 | 0.7889 | 0.8369 | 0.8376 | 0.4315 |
| round2 | F6_icd_ce_lambda025 | valid_acc2_nonzero | 15 | 0.7417 | 0.7169 | 0.7905 | 0.8476 | 0.8484 | 0.4475 |
| round2 | F6_icd_ce_lambda025 | valid_mae | 15 | 0.7417 | 0.7169 | 0.7905 | 0.8476 | 0.8484 | 0.4475 |
| round2 | F7_icd_no_ce_lambda025 | valid_acc2_nonzero | 28 | 0.7592 | 0.6858 | 0.8057 | 0.8521 | 0.8524 | 0.4781 |
| round2 | F7_icd_no_ce_lambda025 | valid_mae | 26 | 0.7464 | 0.6812 | 0.8101 | 0.8613 | 0.8615 | 0.4636 |
| round2 | GP_icd_ce | valid_acc2_nonzero | 16 | 0.7366 | 0.7252 | 0.7940 | 0.8491 | 0.8490 | 0.4329 |
| round2 | GP_icd_ce | valid_mae | 16 | 0.7366 | 0.7252 | 0.7940 | 0.8491 | 0.8490 | 0.4329 |
| round2 | GP_icd_no_ce | valid_acc2_nonzero | 4 | 0.8084 | 0.8122 | 0.7659 | 0.8262 | 0.8271 | 0.4023 |
| round2 | GP_icd_no_ce | valid_mae | 13 | 0.7431 | 0.7746 | 0.8027 | 0.8384 | 0.8394 | 0.4023 |
| round2 | S1_warmup_cosine | valid_acc2_nonzero | 17 | 0.8041 | 0.7176 | 0.7938 | 0.8506 | 0.8503 | 0.4257 |
| round2 | S1_warmup_cosine | valid_mae | 18 | 0.7636 | 0.7004 | 0.8081 | 0.8613 | 0.8620 | 0.4490 |
