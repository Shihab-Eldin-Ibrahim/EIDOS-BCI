# EIDOS-BCI

A modular EEG signal processing and machine learning framework for
Brain–Computer Interface (BCI) research.

## Overview

EIDOS-BCI is a Python-based framework for processing and analyzing EEG
signals using MNE-Python and machine learning techniques.

The project is being developed as part of **IEEE ESPC 2026** and focuses on
Motor Imagery (MI) EEG analysis using the **BCI Competition IV Dataset 2a**.

The project investigates EEG spatial filtering and classification methods
for four-class Motor Imagery BCI.

The long-term goal is to develop a foundation for:

- Motor Imagery Brain–Computer Interfaces
- EEG signal processing
- Machine learning for BCI
- Cross-subject BCI generalization
- Real-time BCI systems
- Brain–Computer Interface and Virtual Reality research

---

## Current Pipeline

The current research pipeline is:

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
              Spatial Filtering
                    │
             ┌──────┴──────┐
             ▼             ▼
          Standard         FBCSP
             CSP
             │             │
             └──────┬──────┘
                    ▼
             Feature Extraction
                    │
                    ▼
              Classification
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         LDA      Logistic    SVM
                  Regression
                    │
             ┌──────┴──────┐
             ▼             ▼
      Within-Subject   Cross-Subject
         Analysis          LOSO
                           │
                           ▼
                 Statistical Analysis
                           │
                           ▼
                 Generalization Analysis
                           │
                           ▼
                    Final Results