# EIDOS-BCI

A modular EEG signal processing framework for Brain–Computer Interface (BCI) research.

## Overview

EIDOS-BCI is a Python-based framework for processing and analyzing EEG signals using MNE-Python.

The project is being developed as part of **IEEE ESPC 2026** and focuses on building a modular pipeline for **Motor Imagery (MI) EEG analysis**.

The long-term goal is to develop a foundation for:

- Motor Imagery Brain–Computer Interfaces
- EEG signal processing
- Machine learning for BCI
- Real-time BCI systems
- Brain–Computer Interface and Virtual Reality research

---

## Current Pipeline

The current processing pipeline is:

```text
BCI Competition IV Dataset 2a
            │
            ▼
       EEG Loading
            │
            ▼
     Dataset Metadata
            │
            ▼
     Event Extraction
            │
            ▼
    Motor Imagery Epoching
            │
            ▼
      8–30 Hz Filtering
            │
            ├───────────────┐
            ▼               ▼
         PSD           ERD / ERS
            │               │
            └───────┬───────┘
                    ▼
                   CSP
                    │
                    ▼
             Classification
               (planned)