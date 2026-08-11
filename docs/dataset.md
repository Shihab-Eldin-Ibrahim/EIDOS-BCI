# BCI Competition IV — Dataset 2a

## Recording

File currently being analyzed:

`A01T.gdf`

## Signal Information

- Total channels: 25
- EEG channels: 22
- EOG channels: 3
- Sampling frequency: 250 Hz
- Duration: approximately 2690 seconds

## Event Types

| Code | Description |
|---:|---|
| 276 | Idle EEG — eyes open |
| 277 | Idle EEG — eyes closed |
| 768 | Start of trial |
| 769 | Left-hand imagery |
| 770 | Right-hand imagery |
| 771 | Foot imagery |
| 772 | Tongue imagery |
| 1023 | Rejected trial |
| 1072 | Eye movements |
| 32766 | Start of new run |

## Current Research Goal

Determine whether EEG signals can be used to classify
motor imagery tasks.

## Next Step

Extract individual trials from the continuous EEG recording
using the event markers.