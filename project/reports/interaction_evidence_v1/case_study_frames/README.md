# Interaction Case Study frames

Each case contains five clean PNG frames sampled at approximately 10%, 30%,
50%, 70%, and 90% of the segmented utterance video. `contact_sheet.png` is a
left-to-right preview of those five frames. `contact_sheet_all.png` stacks the
four cases in the order listed below.

| Case | Frame files | Segment-relative times (s) | Original-video times (s) |
|---|---|---|---|
| `125676[7]` | `125676_7/frame_01.png` ... `frame_05.png` | 0.367, 1.100, 1.867, 2.600, 3.333 | 51.372, 52.105, 52.872, 53.605, 54.338 |
| `130456[11]` | `130456_11/frame_01.png` ... `frame_05.png` | 0.733, 2.167, 3.600, 5.033, 6.467 | 34.438, 35.872, 37.305, 38.738, 40.172 |
| `4dV3slGSc60[8]` | `4dV3slGSc60_8/frame_01.png` ... `frame_05.png` | 0.867, 2.667, 4.400, 6.200, 7.967 | 71.199, 72.999, 74.732, 76.532, 78.299 |
| `70280[8]` | `70280_8/frame_01.png` ... `frame_05.png` | 1.433, 4.300, 7.200, 10.067, 12.933 | 34.360, 37.227, 40.127, 42.994, 45.860 |

Source videos are the silent, utterance-level MOSEI segments under
`dataset/cmu_mosei/media/video_silent/`. The original-video timestamps are
computed by adding the utterance start time to the relative frame time.
