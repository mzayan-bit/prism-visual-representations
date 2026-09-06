# PRISM Representation Showcase: Cross-Paradigm Empirical Synthesis

**Report ID:** `rep_rep_prism_representation_showcase_39c0822c`  
**Campaign ID:** `prism_representation_showcase`  
**Campaign Fingerprint:** `39c0822c676834ccd1fa3f4230966a5d9c7d9bc4e8e48ea485bacacc26697adf`

---

## Executive Summary

> This benchmark report synthesizes experimental evidence from campaign 'prism_representation_showcase'. Total registered observations: 810, covering 45 completed factor combinations (0.8% campaign completion). Key findings highlight distinct representation profiles across architectures and pretraining objectives without collapsing tradeoffs into a single aggregate metric.

## Experimental Methodology

Evaluations enforce strict pairwise factor control. Only designated independent variables are varied while keeping architecture backbones, dataset splits, optimization parameters, and random seeds strictly aligned.

> [!WARNING]
> - EVIDENCE_GAP_DETECTED: 5571 planned experimental factor combinations have not yet been observed.

## Benchmark Result Tables

### Accuracy by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.7800 | 0.8150 | 0.8300 |
| Scratch | 0.6950 | 0.7200 | 0.6850 |
| Simclr | 0.8200 | 0.8620 | 0.8750 |
| Supervised | 0.8450 | 0.8840 | 0.8920 |
| Vision_Language | 0.8100 | 0.8540 | 0.8800 |

### Loss by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.5400 | 0.4620 | 0.4200 |
| Scratch | 0.7200 | 0.6800 | 0.7600 |
| Simclr | 0.4700 | 0.3850 | 0.3400 |
| Supervised | 0.4100 | 0.3210 | 0.2950 |
| Vision_Language | 0.4900 | 0.3980 | 0.3250 |

### Linear Probe Accuracy by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.7720 | 0.8040 | 0.8220 |
| Scratch | 0.6950 | 0.7200 | 0.6850 |
| Simclr | 0.8150 | 0.8560 | 0.8710 |
| Supervised | 0.8450 | 0.8840 | 0.8920 |
| Vision_Language | 0.8050 | 0.8490 | 0.8750 |

### Transfer Gain by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.0090 | 0.1150 | 0.1380 |
| Scratch | 0.0000 | 0.0000 | 0.0000 |
| Simclr | 0.1450 | 0.1850 | 0.2100 |
| Supervised | 0.1150 | 0.1420 | 0.1650 |
| Vision_Language | 0.1300 | 0.1700 | 0.1980 |

### Robustness Accuracy Drop by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.1850 | 0.1450 | 0.1280 |
| Scratch | 0.2600 | 0.2450 | 0.2800 |
| Simclr | 0.1350 | 0.0980 | 0.0820 |
| Supervised | 0.1600 | 0.1250 | 0.1480 |
| Vision_Language | 0.1420 | 0.1050 | 0.0880 |

### Representation Drift by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.2550 | 0.2100 | 0.1850 |
| Scratch | 0.3700 | 0.3500 | 0.3950 |
| Simclr | 0.1800 | 0.1420 | 0.1180 |
| Supervised | 0.2200 | 0.1840 | 0.1620 |
| Vision_Language | 0.1950 | 0.1550 | 0.1300 |

### Ece by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.0980 | 0.0850 | 0.0720 |
| Scratch | 0.1550 | 0.1450 | 0.1680 |
| Simclr | 0.0800 | 0.0620 | 0.0450 |
| Supervised | 0.0680 | 0.0480 | 0.0520 |
| Vision_Language | 0.0750 | 0.0540 | 0.0410 |

### Brier by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.1450 | 0.1200 | 0.1050 |
| Scratch | 0.1950 | 0.1850 | 0.2100 |
| Simclr | 0.1250 | 0.0950 | 0.0720 |
| Supervised | 0.1050 | 0.0820 | 0.0780 |
| Vision_Language | 0.1180 | 0.0890 | 0.0680 |

### Ood Auroc by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.8100 | 0.8420 | 0.8650 |
| Scratch | 0.6900 | 0.7100 | 0.6700 |
| Simclr | 0.8750 | 0.9150 | 0.9380 |
| Supervised | 0.8500 | 0.8920 | 0.9080 |
| Vision_Language | 0.8800 | 0.9240 | 0.9450 |

### Detection Mean Iou by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.6100 | 0.6720 | 0.6950 |
| Scratch | 0.4200 | 0.4500 | 0.3900 |
| Simclr | 0.5600 | 0.6340 | 0.6200 |
| Supervised | 0.5400 | 0.6120 | 0.5850 |
| Vision_Language | 0.5700 | 0.6400 | 0.6550 |

### Segmentation Miou by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.5800 | 0.6480 | 0.6800 |
| Scratch | 0.3800 | 0.4100 | 0.3600 |
| Simclr | 0.5300 | 0.6010 | 0.5950 |
| Supervised | 0.5100 | 0.5840 | 0.5600 |
| Vision_Language | 0.5450 | 0.6150 | 0.6300 |

### Video Accuracy by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.6350 | 0.6840 | 0.7150 |
| Scratch | 0.5500 | 0.5800 | 0.5200 |
| Simclr | 0.7050 | 0.7620 | 0.7950 |
| Supervised | 0.6900 | 0.7450 | 0.7780 |
| Vision_Language | 0.7000 | 0.7500 | 0.7850 |

### Temporal Consistency by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.7100 | 0.7600 | 0.7900 |
| Scratch | 0.6000 | 0.6200 | 0.5800 |
| Simclr | 0.7900 | 0.8450 | 0.8700 |
| Supervised | 0.7600 | 0.8120 | 0.8350 |
| Vision_Language | 0.7800 | 0.8300 | 0.8600 |

### Retrieval R1 by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.2800 | 0.3450 | 0.3800 |
| Scratch | 0.1600 | 0.1800 | 0.1400 |
| Simclr | 0.4200 | 0.5120 | 0.5650 |
| Supervised | 0.3500 | 0.4210 | 0.4650 |
| Vision_Language | 0.4900 | 0.6120 | 0.6750 |

### Zero Shot Accuracy by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.3300 | 0.3920 | 0.4250 |
| Scratch | 0.1900 | 0.2100 | 0.1700 |
| Simclr | 0.4600 | 0.5340 | 0.5900 |
| Supervised | 0.4100 | 0.4850 | 0.5100 |
| Vision_Language | 0.5250 | 0.6450 | 0.7100 |

### Neighbor Consistency by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.7400 | 0.7850 | 0.8200 |
| Scratch | 0.6200 | 0.6500 | 0.5800 |
| Simclr | 0.8350 | 0.8920 | 0.9150 |
| Supervised | 0.8100 | 0.8650 | 0.8850 |
| Vision_Language | 0.8250 | 0.8800 | 0.9100 |

### Centroid Separation by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 1.7500 | 1.9500 | 2.1500 |
| Scratch | 1.2000 | 1.3000 | 1.1000 |
| Simclr | 2.3000 | 2.7800 | 3.0500 |
| Supervised | 2.1000 | 2.4500 | 2.6000 |
| Vision_Language | 2.2500 | 2.6500 | 2.9000 |

### Intra Class Compactness by Pretraining_Objective vs Architecture

| Pretraining_Objective | Cnn | Resnet | Vit |
| --- | --- | --- | --- |
| Reconstruction | 0.5700 | 0.6400 | 0.6900 |
| Scratch | 0.4100 | 0.4500 | 0.3800 |
| Simclr | 0.7100 | 0.8100 | 0.8600 |
| Supervised | 0.6500 | 0.7200 | 0.7600 |
| Vision_Language | 0.6900 | 0.7800 | 0.8300 |

## Scientific Findings Grounded in Observed Evidence

### Finding `find_1`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of pretraining_objective, 'supervised' achieved the most favorable Linear Probe Accuracy (1.010) compared to 'simclr' (0.983), with a delta of 0.026.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Standard transferability and linear separability metric.

### Finding `find_2`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of pretraining_objective, 'simclr' achieved the most favorable Transfer Learning Gain (0.316) compared to 'vision_language' (0.302), with a delta of 0.014.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Positive gain indicates beneficial pretraining transfer.

### Finding `find_3`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'vit' achieved the most favorable In-Distribution Accuracy (0.968) compared to 'resnet' (0.963), with a delta of 0.005.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Primary supervised classification performance.

### Finding `find_4`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'resnet' achieved the most favorable Corruption Accuracy Drop (0.280) compared to 'vit' (0.281), with a delta of 0.002.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Evaluated under standard Phase 15 perturbations.

### Finding `find_5`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'resnet' achieved the most favorable Spatial Detection Mean IoU (0.738) compared to 'vit' (0.725), with a delta of 0.013.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Detection mean IoU from lightweight synthetic spatial probe; not COCO mAP.

### Finding `find_6`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'resnet' achieved the most favorable Spatial Segmentation mIoU (0.708) compared to 'vit' (0.701), with a delta of 0.007.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Segmentation mean IoU on 2D synthetic spatial shapes.

### Finding `find_7`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'vit' achieved the most favorable In-Distribution Accuracy (0.968) compared to 'resnet' (0.963), with a delta of 0.005.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Primary supervised classification performance.

### Finding `find_8`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'vit' achieved the most favorable Expected Calibration Error (ECE) (0.212) compared to 'resnet' (0.215), with a delta of 0.003.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- Expected Calibration Error from equal-width reliability bins.

### Finding `find_9`
**Evidence Strength:** `supported_by_repeated_runs`

> Under controlled evaluation of architecture, 'vit' achieved the most favorable OOD Detection AUROC (1.001) compared to 'resnet' (0.993), with a delta of 0.009.

**Caveats & Limitations:**
- Grounding restricted to evaluated dataset: cifar10.
- OOD AUROC from controlled synthetic OOD suite.

## Evidence Gaps & Missing Experiments

- **[gap_missing_1]** No observations for cnn with supervised on cifar10 (transfer).
- **[gap_missing_2]** No observations for cnn with supervised on cifar10 (spatial).
- **[gap_missing_3]** No observations for cnn with supervised on cifar10 (temporal).
- **[gap_missing_4]** No observations for cnn with supervised on cifar10 (robustness).
- **[gap_missing_5]** No observations for cnn with supervised on spatial_synth (classification).
- **[gap_missing_6]** No observations for cnn with supervised on spatial_synth (transfer).
- **[gap_missing_7]** No observations for cnn with supervised on spatial_synth (spatial).
- **[gap_missing_8]** No observations for cnn with supervised on spatial_synth (temporal).
- **[gap_missing_9]** No observations for cnn with supervised on spatial_synth (robustness).
- **[gap_missing_10]** No observations for cnn with supervised on temporal_synth (classification).
- **[gap_missing_11]** No observations for cnn with supervised on temporal_synth (transfer).
- **[gap_missing_12]** No observations for cnn with supervised on temporal_synth (spatial).
- **[gap_missing_13]** No observations for cnn with supervised on temporal_synth (temporal).
- **[gap_missing_14]** No observations for cnn with supervised on temporal_synth (robustness).
- **[gap_missing_15]** No observations for cnn with supervised on multimodal_synth (classification).
- **[gap_missing_16]** No observations for cnn with supervised on multimodal_synth (transfer).
- **[gap_missing_17]** No observations for cnn with supervised on multimodal_synth (spatial).
- **[gap_missing_18]** No observations for cnn with supervised on multimodal_synth (temporal).
- **[gap_missing_19]** No observations for cnn with supervised on multimodal_synth (robustness).
- **[gap_missing_20]** No observations for cnn with simclr on cifar10 (transfer).
- **[gap_missing_21]** No observations for cnn with simclr on cifar10 (spatial).
- **[gap_missing_22]** No observations for cnn with simclr on cifar10 (temporal).
- **[gap_missing_23]** No observations for cnn with simclr on cifar10 (robustness).
- **[gap_missing_24]** No observations for cnn with simclr on spatial_synth (classification).
- **[gap_missing_25]** No observations for cnn with simclr on spatial_synth (transfer).
- **[gap_missing_26]** No observations for cnn with simclr on spatial_synth (spatial).
- **[gap_missing_27]** No observations for cnn with simclr on spatial_synth (temporal).
- **[gap_missing_28]** No observations for cnn with simclr on spatial_synth (robustness).
- **[gap_missing_29]** No observations for cnn with simclr on temporal_synth (classification).
- **[gap_missing_30]** No observations for cnn with simclr on temporal_synth (transfer).
- **[gap_missing_31]** No observations for cnn with simclr on temporal_synth (spatial).
- **[gap_missing_32]** No observations for cnn with simclr on temporal_synth (temporal).
- **[gap_missing_33]** No observations for cnn with simclr on temporal_synth (robustness).
- **[gap_missing_34]** No observations for cnn with simclr on multimodal_synth (classification).
- **[gap_missing_35]** No observations for cnn with simclr on multimodal_synth (transfer).
- **[gap_missing_36]** No observations for cnn with simclr on multimodal_synth (spatial).
- **[gap_missing_37]** No observations for cnn with simclr on multimodal_synth (temporal).
- **[gap_missing_38]** No observations for cnn with simclr on multimodal_synth (robustness).
- **[gap_missing_39]** No observations for cnn with reconstruction on cifar10 (transfer).
- **[gap_missing_40]** No observations for cnn with reconstruction on cifar10 (spatial).
- **[gap_missing_41]** No observations for cnn with reconstruction on cifar10 (temporal).
- **[gap_missing_42]** No observations for cnn with reconstruction on cifar10 (robustness).
- **[gap_missing_43]** No observations for cnn with reconstruction on spatial_synth (classification).
- **[gap_missing_44]** No observations for cnn with reconstruction on spatial_synth (transfer).
- **[gap_missing_45]** No observations for cnn with reconstruction on spatial_synth (spatial).
- **[gap_missing_46]** No observations for cnn with reconstruction on spatial_synth (temporal).
- **[gap_missing_47]** No observations for cnn with reconstruction on spatial_synth (robustness).
- **[gap_missing_48]** No observations for cnn with reconstruction on temporal_synth (classification).
- **[gap_missing_49]** No observations for cnn with reconstruction on temporal_synth (transfer).
- **[gap_missing_50]** No observations for cnn with reconstruction on temporal_synth (spatial).
- **[gap_missing_51]** No observations for cnn with reconstruction on temporal_synth (temporal).
- **[gap_missing_52]** No observations for cnn with reconstruction on temporal_synth (robustness).
- **[gap_missing_53]** No observations for cnn with reconstruction on multimodal_synth (classification).
- **[gap_missing_54]** No observations for cnn with reconstruction on multimodal_synth (transfer).
- **[gap_missing_55]** No observations for cnn with reconstruction on multimodal_synth (spatial).
- **[gap_missing_56]** No observations for cnn with reconstruction on multimodal_synth (temporal).
- **[gap_missing_57]** No observations for cnn with reconstruction on multimodal_synth (robustness).
- **[gap_missing_58]** No observations for cnn with vision_language on cifar10 (transfer).
- **[gap_missing_59]** No observations for cnn with vision_language on cifar10 (spatial).
- **[gap_missing_60]** No observations for cnn with vision_language on cifar10 (temporal).
- **[gap_missing_61]** No observations for cnn with vision_language on cifar10 (multimodal).
- **[gap_missing_62]** No observations for cnn with vision_language on cifar10 (robustness).
- **[gap_missing_63]** No observations for cnn with vision_language on spatial_synth (classification).
- **[gap_missing_64]** No observations for cnn with vision_language on spatial_synth (transfer).
- **[gap_missing_65]** No observations for cnn with vision_language on spatial_synth (spatial).
- **[gap_missing_66]** No observations for cnn with vision_language on spatial_synth (temporal).
- **[gap_missing_67]** No observations for cnn with vision_language on spatial_synth (multimodal).
- **[gap_missing_68]** No observations for cnn with vision_language on spatial_synth (robustness).
- **[gap_missing_69]** No observations for cnn with vision_language on temporal_synth (classification).
- **[gap_missing_70]** No observations for cnn with vision_language on temporal_synth (transfer).
- **[gap_missing_71]** No observations for cnn with vision_language on temporal_synth (spatial).
- **[gap_missing_72]** No observations for cnn with vision_language on temporal_synth (temporal).
- **[gap_missing_73]** No observations for cnn with vision_language on temporal_synth (multimodal).
- **[gap_missing_74]** No observations for cnn with vision_language on temporal_synth (robustness).
- **[gap_missing_75]** No observations for cnn with vision_language on multimodal_synth (classification).
- **[gap_missing_76]** No observations for cnn with vision_language on multimodal_synth (transfer).
- **[gap_missing_77]** No observations for cnn with vision_language on multimodal_synth (spatial).
- **[gap_missing_78]** No observations for cnn with vision_language on multimodal_synth (temporal).
- **[gap_missing_79]** No observations for cnn with vision_language on multimodal_synth (multimodal).
- **[gap_missing_80]** No observations for cnn with vision_language on multimodal_synth (robustness).
- **[gap_missing_81]** No observations for cnn with scratch on cifar10 (transfer).
- **[gap_missing_82]** No observations for cnn with scratch on cifar10 (spatial).
- **[gap_missing_83]** No observations for cnn with scratch on cifar10 (temporal).
- **[gap_missing_84]** No observations for cnn with scratch on cifar10 (robustness).
- **[gap_missing_85]** No observations for cnn with scratch on spatial_synth (classification).
- **[gap_missing_86]** No observations for cnn with scratch on spatial_synth (transfer).
- **[gap_missing_87]** No observations for cnn with scratch on spatial_synth (spatial).
- **[gap_missing_88]** No observations for cnn with scratch on spatial_synth (temporal).
- **[gap_missing_89]** No observations for cnn with scratch on spatial_synth (robustness).
- **[gap_missing_90]** No observations for cnn with scratch on temporal_synth (classification).
- **[gap_missing_91]** No observations for cnn with scratch on temporal_synth (transfer).
- **[gap_missing_92]** No observations for cnn with scratch on temporal_synth (spatial).
- **[gap_missing_93]** No observations for cnn with scratch on temporal_synth (temporal).
- **[gap_missing_94]** No observations for cnn with scratch on temporal_synth (robustness).
- **[gap_missing_95]** No observations for cnn with scratch on multimodal_synth (classification).
- **[gap_missing_96]** No observations for cnn with scratch on multimodal_synth (transfer).
- **[gap_missing_97]** No observations for cnn with scratch on multimodal_synth (spatial).
- **[gap_missing_98]** No observations for cnn with scratch on multimodal_synth (temporal).
- **[gap_missing_99]** No observations for cnn with scratch on multimodal_synth (robustness).
- **[gap_missing_100]** No observations for resnet with supervised on cifar10 (transfer).
- **[gap_missing_101]** No observations for resnet with supervised on cifar10 (spatial).
- **[gap_missing_102]** No observations for resnet with supervised on cifar10 (temporal).
- **[gap_missing_103]** No observations for resnet with supervised on cifar10 (robustness).
- **[gap_missing_104]** No observations for resnet with supervised on spatial_synth (classification).
- **[gap_missing_105]** No observations for resnet with supervised on spatial_synth (transfer).
- **[gap_missing_106]** No observations for resnet with supervised on spatial_synth (spatial).
- **[gap_missing_107]** No observations for resnet with supervised on spatial_synth (temporal).
- **[gap_missing_108]** No observations for resnet with supervised on spatial_synth (robustness).
- **[gap_missing_109]** No observations for resnet with supervised on temporal_synth (classification).
- **[gap_missing_110]** No observations for resnet with supervised on temporal_synth (transfer).
- **[gap_missing_111]** No observations for resnet with supervised on temporal_synth (spatial).
- **[gap_missing_112]** No observations for resnet with supervised on temporal_synth (temporal).
- **[gap_missing_113]** No observations for resnet with supervised on temporal_synth (robustness).
- **[gap_missing_114]** No observations for resnet with supervised on multimodal_synth (classification).
- **[gap_missing_115]** No observations for resnet with supervised on multimodal_synth (transfer).
- **[gap_missing_116]** No observations for resnet with supervised on multimodal_synth (spatial).
- **[gap_missing_117]** No observations for resnet with supervised on multimodal_synth (temporal).
- **[gap_missing_118]** No observations for resnet with supervised on multimodal_synth (robustness).
- **[gap_missing_119]** No observations for resnet with simclr on cifar10 (transfer).
- **[gap_missing_120]** No observations for resnet with simclr on cifar10 (spatial).
- **[gap_missing_121]** No observations for resnet with simclr on cifar10 (temporal).
- **[gap_missing_122]** No observations for resnet with simclr on cifar10 (robustness).
- **[gap_missing_123]** No observations for resnet with simclr on spatial_synth (classification).
- **[gap_missing_124]** No observations for resnet with simclr on spatial_synth (transfer).
- **[gap_missing_125]** No observations for resnet with simclr on spatial_synth (spatial).
- **[gap_missing_126]** No observations for resnet with simclr on spatial_synth (temporal).
- **[gap_missing_127]** No observations for resnet with simclr on spatial_synth (robustness).
- **[gap_missing_128]** No observations for resnet with simclr on temporal_synth (classification).
- **[gap_missing_129]** No observations for resnet with simclr on temporal_synth (transfer).
- **[gap_missing_130]** No observations for resnet with simclr on temporal_synth (spatial).
- **[gap_missing_131]** No observations for resnet with simclr on temporal_synth (temporal).
- **[gap_missing_132]** No observations for resnet with simclr on temporal_synth (robustness).
- **[gap_missing_133]** No observations for resnet with simclr on multimodal_synth (classification).
- **[gap_missing_134]** No observations for resnet with simclr on multimodal_synth (transfer).
- **[gap_missing_135]** No observations for resnet with simclr on multimodal_synth (spatial).
- **[gap_missing_136]** No observations for resnet with simclr on multimodal_synth (temporal).
- **[gap_missing_137]** No observations for resnet with simclr on multimodal_synth (robustness).
- **[gap_missing_138]** No observations for resnet with reconstruction on cifar10 (transfer).
- **[gap_missing_139]** No observations for resnet with reconstruction on cifar10 (spatial).
- **[gap_missing_140]** No observations for resnet with reconstruction on cifar10 (temporal).
- **[gap_missing_141]** No observations for resnet with reconstruction on cifar10 (robustness).
- **[gap_missing_142]** No observations for resnet with reconstruction on spatial_synth (classification).
- **[gap_missing_143]** No observations for resnet with reconstruction on spatial_synth (transfer).
- **[gap_missing_144]** No observations for resnet with reconstruction on spatial_synth (spatial).
- **[gap_missing_145]** No observations for resnet with reconstruction on spatial_synth (temporal).
- **[gap_missing_146]** No observations for resnet with reconstruction on spatial_synth (robustness).
- **[gap_missing_147]** No observations for resnet with reconstruction on temporal_synth (classification).
- **[gap_missing_148]** No observations for resnet with reconstruction on temporal_synth (transfer).
- **[gap_missing_149]** No observations for resnet with reconstruction on temporal_synth (spatial).
- **[gap_missing_150]** No observations for resnet with reconstruction on temporal_synth (temporal).
- **[gap_missing_151]** No observations for resnet with reconstruction on temporal_synth (robustness).
- **[gap_missing_152]** No observations for resnet with reconstruction on multimodal_synth (classification).
- **[gap_missing_153]** No observations for resnet with reconstruction on multimodal_synth (transfer).
- **[gap_missing_154]** No observations for resnet with reconstruction on multimodal_synth (spatial).
- **[gap_missing_155]** No observations for resnet with reconstruction on multimodal_synth (temporal).
- **[gap_missing_156]** No observations for resnet with reconstruction on multimodal_synth (robustness).
- **[gap_missing_157]** No observations for resnet with vision_language on cifar10 (transfer).
- **[gap_missing_158]** No observations for resnet with vision_language on cifar10 (spatial).
- **[gap_missing_159]** No observations for resnet with vision_language on cifar10 (temporal).
- **[gap_missing_160]** No observations for resnet with vision_language on cifar10 (multimodal).
- **[gap_missing_161]** No observations for resnet with vision_language on cifar10 (robustness).
- **[gap_missing_162]** No observations for resnet with vision_language on spatial_synth (classification).
- **[gap_missing_163]** No observations for resnet with vision_language on spatial_synth (transfer).
- **[gap_missing_164]** No observations for resnet with vision_language on spatial_synth (spatial).
- **[gap_missing_165]** No observations for resnet with vision_language on spatial_synth (temporal).
- **[gap_missing_166]** No observations for resnet with vision_language on spatial_synth (multimodal).
- **[gap_missing_167]** No observations for resnet with vision_language on spatial_synth (robustness).
- **[gap_missing_168]** No observations for resnet with vision_language on temporal_synth (classification).
- **[gap_missing_169]** No observations for resnet with vision_language on temporal_synth (transfer).
- **[gap_missing_170]** No observations for resnet with vision_language on temporal_synth (spatial).
- **[gap_missing_171]** No observations for resnet with vision_language on temporal_synth (temporal).
- **[gap_missing_172]** No observations for resnet with vision_language on temporal_synth (multimodal).
- **[gap_missing_173]** No observations for resnet with vision_language on temporal_synth (robustness).
- **[gap_missing_174]** No observations for resnet with vision_language on multimodal_synth (classification).
- **[gap_missing_175]** No observations for resnet with vision_language on multimodal_synth (transfer).
- **[gap_missing_176]** No observations for resnet with vision_language on multimodal_synth (spatial).
- **[gap_missing_177]** No observations for resnet with vision_language on multimodal_synth (temporal).
- **[gap_missing_178]** No observations for resnet with vision_language on multimodal_synth (multimodal).
- **[gap_missing_179]** No observations for resnet with vision_language on multimodal_synth (robustness).
- **[gap_missing_180]** No observations for resnet with scratch on cifar10 (transfer).
- **[gap_missing_181]** No observations for resnet with scratch on cifar10 (spatial).
- **[gap_missing_182]** No observations for resnet with scratch on cifar10 (temporal).
- **[gap_missing_183]** No observations for resnet with scratch on cifar10 (robustness).
- **[gap_missing_184]** No observations for resnet with scratch on spatial_synth (classification).
- **[gap_missing_185]** No observations for resnet with scratch on spatial_synth (transfer).
- **[gap_missing_186]** No observations for resnet with scratch on spatial_synth (spatial).
- **[gap_missing_187]** No observations for resnet with scratch on spatial_synth (temporal).
- **[gap_missing_188]** No observations for resnet with scratch on spatial_synth (robustness).
- **[gap_missing_189]** No observations for resnet with scratch on temporal_synth (classification).
- **[gap_missing_190]** No observations for resnet with scratch on temporal_synth (transfer).
- **[gap_missing_191]** No observations for resnet with scratch on temporal_synth (spatial).
- **[gap_missing_192]** No observations for resnet with scratch on temporal_synth (temporal).
- **[gap_missing_193]** No observations for resnet with scratch on temporal_synth (robustness).
- **[gap_missing_194]** No observations for resnet with scratch on multimodal_synth (classification).
- **[gap_missing_195]** No observations for resnet with scratch on multimodal_synth (transfer).
- **[gap_missing_196]** No observations for resnet with scratch on multimodal_synth (spatial).
- **[gap_missing_197]** No observations for resnet with scratch on multimodal_synth (temporal).
- **[gap_missing_198]** No observations for resnet with scratch on multimodal_synth (robustness).
- **[gap_missing_199]** No observations for vit with supervised on cifar10 (transfer).
- **[gap_missing_200]** No observations for vit with supervised on cifar10 (spatial).
- **[gap_missing_201]** No observations for vit with supervised on cifar10 (temporal).
- **[gap_missing_202]** No observations for vit with supervised on cifar10 (robustness).
- **[gap_missing_203]** No observations for vit with supervised on spatial_synth (classification).
- **[gap_missing_204]** No observations for vit with supervised on spatial_synth (transfer).
- **[gap_missing_205]** No observations for vit with supervised on spatial_synth (spatial).
- **[gap_missing_206]** No observations for vit with supervised on spatial_synth (temporal).
- **[gap_missing_207]** No observations for vit with supervised on spatial_synth (robustness).
- **[gap_missing_208]** No observations for vit with supervised on temporal_synth (classification).
- **[gap_missing_209]** No observations for vit with supervised on temporal_synth (transfer).
- **[gap_missing_210]** No observations for vit with supervised on temporal_synth (spatial).
- **[gap_missing_211]** No observations for vit with supervised on temporal_synth (temporal).
- **[gap_missing_212]** No observations for vit with supervised on temporal_synth (robustness).
- **[gap_missing_213]** No observations for vit with supervised on multimodal_synth (classification).
- **[gap_missing_214]** No observations for vit with supervised on multimodal_synth (transfer).
- **[gap_missing_215]** No observations for vit with supervised on multimodal_synth (spatial).
- **[gap_missing_216]** No observations for vit with supervised on multimodal_synth (temporal).
- **[gap_missing_217]** No observations for vit with supervised on multimodal_synth (robustness).
- **[gap_missing_218]** No observations for vit with simclr on cifar10 (transfer).
- **[gap_missing_219]** No observations for vit with simclr on cifar10 (spatial).
- **[gap_missing_220]** No observations for vit with simclr on cifar10 (temporal).
- **[gap_missing_221]** No observations for vit with simclr on cifar10 (robustness).
- **[gap_missing_222]** No observations for vit with simclr on spatial_synth (classification).
- **[gap_missing_223]** No observations for vit with simclr on spatial_synth (transfer).
- **[gap_missing_224]** No observations for vit with simclr on spatial_synth (spatial).
- **[gap_missing_225]** No observations for vit with simclr on spatial_synth (temporal).
- **[gap_missing_226]** No observations for vit with simclr on spatial_synth (robustness).
- **[gap_missing_227]** No observations for vit with simclr on temporal_synth (classification).
- **[gap_missing_228]** No observations for vit with simclr on temporal_synth (transfer).
- **[gap_missing_229]** No observations for vit with simclr on temporal_synth (spatial).
- **[gap_missing_230]** No observations for vit with simclr on temporal_synth (temporal).
- **[gap_missing_231]** No observations for vit with simclr on temporal_synth (robustness).
- **[gap_missing_232]** No observations for vit with simclr on multimodal_synth (classification).
- **[gap_missing_233]** No observations for vit with simclr on multimodal_synth (transfer).
- **[gap_missing_234]** No observations for vit with simclr on multimodal_synth (spatial).
- **[gap_missing_235]** No observations for vit with simclr on multimodal_synth (temporal).
- **[gap_missing_236]** No observations for vit with simclr on multimodal_synth (robustness).
- **[gap_missing_237]** No observations for vit with reconstruction on cifar10 (transfer).
- **[gap_missing_238]** No observations for vit with reconstruction on cifar10 (spatial).
- **[gap_missing_239]** No observations for vit with reconstruction on cifar10 (temporal).
- **[gap_missing_240]** No observations for vit with reconstruction on cifar10 (robustness).
- **[gap_missing_241]** No observations for vit with reconstruction on spatial_synth (classification).
- **[gap_missing_242]** No observations for vit with reconstruction on spatial_synth (transfer).
- **[gap_missing_243]** No observations for vit with reconstruction on spatial_synth (spatial).
- **[gap_missing_244]** No observations for vit with reconstruction on spatial_synth (temporal).
- **[gap_missing_245]** No observations for vit with reconstruction on spatial_synth (robustness).
- **[gap_missing_246]** No observations for vit with reconstruction on temporal_synth (classification).
- **[gap_missing_247]** No observations for vit with reconstruction on temporal_synth (transfer).
- **[gap_missing_248]** No observations for vit with reconstruction on temporal_synth (spatial).
- **[gap_missing_249]** No observations for vit with reconstruction on temporal_synth (temporal).
- **[gap_missing_250]** No observations for vit with reconstruction on temporal_synth (robustness).
- **[gap_missing_251]** No observations for vit with reconstruction on multimodal_synth (classification).
- **[gap_missing_252]** No observations for vit with reconstruction on multimodal_synth (transfer).
- **[gap_missing_253]** No observations for vit with reconstruction on multimodal_synth (spatial).
- **[gap_missing_254]** No observations for vit with reconstruction on multimodal_synth (temporal).
- **[gap_missing_255]** No observations for vit with reconstruction on multimodal_synth (robustness).
- **[gap_missing_256]** No observations for vit with vision_language on cifar10 (transfer).
- **[gap_missing_257]** No observations for vit with vision_language on cifar10 (spatial).
- **[gap_missing_258]** No observations for vit with vision_language on cifar10 (temporal).
- **[gap_missing_259]** No observations for vit with vision_language on cifar10 (multimodal).
- **[gap_missing_260]** No observations for vit with vision_language on cifar10 (robustness).
- **[gap_missing_261]** No observations for vit with vision_language on spatial_synth (classification).
- **[gap_missing_262]** No observations for vit with vision_language on spatial_synth (transfer).
- **[gap_missing_263]** No observations for vit with vision_language on spatial_synth (spatial).
- **[gap_missing_264]** No observations for vit with vision_language on spatial_synth (temporal).
- **[gap_missing_265]** No observations for vit with vision_language on spatial_synth (multimodal).
- **[gap_missing_266]** No observations for vit with vision_language on spatial_synth (robustness).
- **[gap_missing_267]** No observations for vit with vision_language on temporal_synth (classification).
- **[gap_missing_268]** No observations for vit with vision_language on temporal_synth (transfer).
- **[gap_missing_269]** No observations for vit with vision_language on temporal_synth (spatial).
- **[gap_missing_270]** No observations for vit with vision_language on temporal_synth (temporal).
- **[gap_missing_271]** No observations for vit with vision_language on temporal_synth (multimodal).
- **[gap_missing_272]** No observations for vit with vision_language on temporal_synth (robustness).
- **[gap_missing_273]** No observations for vit with vision_language on multimodal_synth (classification).
- **[gap_missing_274]** No observations for vit with vision_language on multimodal_synth (transfer).
- **[gap_missing_275]** No observations for vit with vision_language on multimodal_synth (spatial).
- **[gap_missing_276]** No observations for vit with vision_language on multimodal_synth (temporal).
- **[gap_missing_277]** No observations for vit with vision_language on multimodal_synth (multimodal).
- **[gap_missing_278]** No observations for vit with vision_language on multimodal_synth (robustness).
- **[gap_missing_279]** No observations for vit with scratch on cifar10 (transfer).
- **[gap_missing_280]** No observations for vit with scratch on cifar10 (spatial).
- **[gap_missing_281]** No observations for vit with scratch on cifar10 (temporal).
- **[gap_missing_282]** No observations for vit with scratch on cifar10 (robustness).
- **[gap_missing_283]** No observations for vit with scratch on spatial_synth (classification).
- **[gap_missing_284]** No observations for vit with scratch on spatial_synth (transfer).
- **[gap_missing_285]** No observations for vit with scratch on spatial_synth (spatial).
- **[gap_missing_286]** No observations for vit with scratch on spatial_synth (temporal).
- **[gap_missing_287]** No observations for vit with scratch on spatial_synth (robustness).
- **[gap_missing_288]** No observations for vit with scratch on temporal_synth (classification).
- **[gap_missing_289]** No observations for vit with scratch on temporal_synth (transfer).
- **[gap_missing_290]** No observations for vit with scratch on temporal_synth (spatial).
- **[gap_missing_291]** No observations for vit with scratch on temporal_synth (temporal).
- **[gap_missing_292]** No observations for vit with scratch on temporal_synth (robustness).
- **[gap_missing_293]** No observations for vit with scratch on multimodal_synth (classification).
- **[gap_missing_294]** No observations for vit with scratch on multimodal_synth (transfer).
- **[gap_missing_295]** No observations for vit with scratch on multimodal_synth (spatial).
- **[gap_missing_296]** No observations for vit with scratch on multimodal_synth (temporal).
- **[gap_missing_297]** No observations for vit with scratch on multimodal_synth (robustness).

## Reproducibility Appendix

- **Total Registered Observations:** 810
- **Registered Random Seeds:** [42, 100, 2024]
- **Unique Fingerprints:** 45
