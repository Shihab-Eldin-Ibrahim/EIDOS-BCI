# EIDOS-BCI Architecture

## Overview

EIDOS-BCI is organized as a modular EEG processing pipeline.

The architecture separates data loading, metadata handling, preprocessing, signal analysis, and machine learning so that each stage can be developed and tested independently.

---

## System Architecture

```text
                    ┌─────────────────────────┐
                    │   BCI Competition IV    │
                    │       Dataset 2a        │
                    │        A01T.gdf          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       EEG Loader         │
                    │      core/loader.py      │
                    │                          │
                    │  Load GDF → MNE Raw      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     EEG Metadata        │
                    │    core/metadata.py     │
                    │                          │
                    │ Channels / Sampling     │
                    │ Events / Duration       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Epoching           │
                    │ preprocessing/epoching.py│
                    │                          │
                    │ Motor Imagery Events     │
                    │ 0–4 second epochs        │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Filtering          │
                    │ preprocessing/filtering.py│
                    │                          │
                    │       8–30 Hz             │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
       ┌─────────────────────┐       ┌─────────────────────┐
       │      PSD Analysis   │       │     ERD / ERS       │
       │                     │       │                     │
       │ Frequency-domain    │       │ Mu: 8–13 Hz         │
       │ power analysis      │       │ Beta: 13–30 Hz      │
       └──────────┬──────────┘       └──────────┬──────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │          CSP            │
                    │      analysis/csp.py    │
                    │                          │
                    │ Spatial filtering and   │
                    │ feature extraction      │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     Machine Learning    │
                    │                         │
                    │         LDA             │
                    │      (Planned)          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │       Evaluation        │
                    │                         │
                    │ Accuracy                │
                    │ Confusion Matrix        │
                    │ Classification Report   │
                    │      (Planned)          │
                    └─────────────────────────┘