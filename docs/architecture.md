# EIDOS-BCI Architecture

## Overview

EIDOS-BCI is a modular EEG signal processing and machine learning framework for
Brain–Computer Interface (BCI) research.

The system is organized into separate layers for:

- EEG data loading
- Dataset metadata and event handling
- Signal preprocessing
- EEG analysis and feature extraction
- Within-subject classification
- Cross-subject evaluation
- Statistical analysis
- Research result generation

The architecture is designed so that individual processing stages can be
developed, tested, and extended independently.

---

## System Architecture

```text
                    ┌──────────────────────────────┐
                    │   BCI Competition IV         │
                    │       Dataset 2a             │
                    │                              │
                    │  9 subjects                  │
                    │  22 EEG + 3 EOG channels    │
                    │  250 Hz                      │
                    │  4 Motor Imagery classes     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │         EEG Loading           │
                    │                              │
                    │       core/loader.py         │
                    │                              │
                    │   GDF → MNE Raw object       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Dataset Metadata         │
                    │                              │
                    │      core/metadata.py        │
                    │                              │
                    │ Channels / Sampling Rate     │
                    │ Events / Recording Info      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Preprocessing          │
                    │                              │
                    │      preprocessing/          │
                    │                              │
                    │ Event Extraction             │
                    │ Motor Imagery Epoching       │
                    │ 8–30 Hz Bandpass Filtering   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
              ┌────────────────────┴────────────────────┐
              │                                         │
              ▼                                         ▼
   ┌──────────────────────┐                  ┌──────────────────────┐
   │     Signal Analysis  │                  │   Spatial Features   │
   │                      │                  │                      │
   │ PSD                  │                  │ Standard CSP         │
   │ ERD / ERS            │                  │ Filter Bank CSP      │
   │                      │                  │                      │
   └──────────┬───────────┘                  └──────────┬───────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Feature Extraction       │
                    │                              │
                    │ CSP / FBCSP feature vectors  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        Classification         │
                    │                              │
                    │ LDA                          │
                    │ SVM                          │
                    │ Logistic Regression           │
                    │                              │
                    │ Four-class MI classification │
                    └──────────────┬───────────────┘
                                   │
                    ┌──────────────┴───────────────┐
                    │                              │
                    ▼                              ▼
        ┌──────────────────────┐       ┌────────────────────────┐
        │  Within-Subject      │       │   Cross-Subject        │
        │     Evaluation       │       │       Evaluation       │
        │                      │       │                        │
        │ Subject-specific     │       │ Leave-One-Subject-Out  │
        │ training/testing     │       │ (LOSO) evaluation       │
        └──────────┬───────────┘       └────────────┬───────────┘
                   │                                │
                   └──────────────┬─────────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────────┐
                    │        Evaluation             │
                    │                              │
                    │ Accuracy                     │
                    │ Confusion Matrices           │
                    │ Subject-level performance    │
                    │ Generalization gap            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │     Statistical Analysis      │
                    │                              │
                    │ Method comparisons            │
                    │ Subject improvements          │
                    │ Statistical comparisons        │
                    │ LOSO statistical analysis      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Research Outputs        │
                    │                              │
                    │ CSV result tables             │
                    │ Paper-ready tables            │
                    │ Performance plots              │
                    │ Confusion matrices             │
                    │ Final research conclusion      │
                    └──────────────────────────────┘