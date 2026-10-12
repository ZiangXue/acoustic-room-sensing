# Literature Survey: Non-Visual, Low-Power Sensing of Conversation Groups

Compiled 2026-10-11 · Based on abstract- and snippet-level searches; full texts not yet read, so please verify before use

## Overview

The project aims to identify conversation subgroups in a room using a microphone array (optionally with UWB) on an embedded device, assign each person a persistent anonymous ID, and compare accuracy and energy cost against a vision baseline.

Two questions guided the search:

1. Are there similar datasets?
2. Is there similar or related prior work?

**Main conclusion:** Within the scope of this search, I did not find a dataset that combines a microphone array, moving people, multiple simultaneous groups, and group-level ground truth. This supports the plan of simulation-first development with a few real validation sessions. On the other hand, conversation grouping from a microphone array plus DOA was already studied around 2010 (Kawaguchi et al.), so the project's value has to come from modern methods, embedded deployment and energy measurement, the comparison with vision, and UWB fusion. The proposal should state this clearly.

## 1. Datasets

### 1.1 Group annotations, but no microphone array

Useful for testing the grouping stage: use speaking status and position as inputs, without the acoustic front end.

| Dataset | Scale and sensors | Annotations | Notes |
| --- | --- | --- | --- |
| [ConfLab](https://data.4tu.nl/collections/66fca409-5f94-48d7-a317-2ffe028e5608/1) (TU Delft, NeurIPS 2022) | 48 people, 8 overhead cameras; chest-worn badge per person: low-frequency audio (1250 Hz), 9-axis IMU, Bluetooth proximity | Per-person speaking status (60 Hz), F-formations, pose keypoints; 16 of 45 recorded minutes annotated | GitHub code and baseline tasks (speaking status, F-formation estimation) |
| [SALSA](https://arxiv.org/pdf/1506.06882) | 18 people, 60+ minutes, 4 cameras; per-person badge with microphone, accelerometer, Bluetooth, infrared | Position, head and body orientation, F-formations | Poster session and cocktail party scenarios |
| [MatchNMingle](https://research.tudelft.nl/en/publications/the-matchnmingle-dataset-a-novel-multi-sensor-resource-for-the-an) | 92 people, about 2 hours; acceleration, binary proximity, video, audio | Positions and groupings manually annotated; speaking and gesture annotations for part of the data | Sources disagree on how many minutes of F-formation annotation exist; verify before use |

### 1.2 Microphone arrays, but only a single conversation group

Useful for testing the acoustic front end (VAD, speaker segmentation, DOA), but not grouping.

- [CHiME-6](https://ar5iv.labs.arxiv.org/html/2004.09249): dinner parties with 4 participants, six 4-microphone Kinect arrays, plus binaural microphones worn by each person; people move between kitchen, dining room and living room.
- [DiPCo](https://arxiv.org/pdf/2407.16447): dinner parties with 4 participants in a single room, 5 devices each with a 7-microphone circular array (6+1), plus lapel microphones.

### 1.3 Multiple simultaneous conversations (closest to this project's task)

- [MCoRec](https://arxiv.org/pdf/2510.23276v1) (CHiME-9 challenge): up to 8 participants in up to 4 simultaneous conversations, with speech overlap up to 100%; the task includes clustering speakers into their conversation groups. Limitation: participants sit around a large table, so positions are essentially fixed. The [participant report](https://arxiv.org/pdf/2603.01415) I saw uses 360-degree video plus single-channel audio; I could not confirm whether array audio is provided.
- Worth noting: the USTC system used LLM-based zero-shot conversation clustering and reached a speaker clustering F1 of 1.0 on the development set (joint ASR and clustering error rate 15.70%). This suggests semantic content is very informative for grouping and supports the stretch goal, but the system relies on ASR (speaker word error rate about 32%), which is heavy for a small device.

### 1.4 Simulation material

- [MYRiAD](https://arxiv.org/pdf/2301.13057): a multi-array room acoustics database that includes cocktail-party background recordings with 10 to 15 people moving and chatting. It can add realistic noise to simulations but has no group annotations.

### 1.5 How to use these datasets

- Acoustic front end: CHiME-6 and DiPCo.
- Grouping algorithm: speaking-status and position annotations from ConfLab, SALSA and MatchNMingle.
- End to end: our own simulated data plus real sessions.
- Access: some datasets require an application or a signed agreement; MCoRec is challenge data, so check its terms of use.

## 2. Related Work

### 2.1 Conversation grouping with microphone arrays (closest; read first)

- **Kawaguchi et al., Interspeech 2010**, ["Turn taking-based conversation detection by using DOA estimation"](https://www.isca-archive.org/interspeech_2010/kawaguchi10_interspeech.html): detects conversation groups when several groups talk at once, using only hands-free microphone arrays in the room. It combines turn-taking-based detection with DOA-based VAD and estimates the number of speakers from turn-taking rules; results are comparable to methods using wearable microphones. This is almost exactly our MVP, but without deep models and without an energy perspective.
- **"Conversation clustering based on PLCA"**, [EUSIPCO 2012](https://new.eurasip.org/Proceedings/Eusipco/Eusipco2012/Conference/papers/1569574751.pdf): combines DOA estimation with mutual-information clustering, extending the approach beyond two-person conversations to groups with more than two speakers.

### 2.2 Acoustic localization and speaker segmentation

- Lathoud's Idiap PhD thesis, [Spatio-Temporal Analysis of Spontaneous Speech with Microphone Arrays](https://publications.idiap.ch/publications/show/308): uses arrays to answer "who spoke where and when," including multiple moving speakers, with short-term clustering of multi-speaker location estimates.
- Lathoud et al., [Eurospeech 2003](https://www.isca-archive.org/eurospeech_2003/lathoud03_eurospeech.html): when several people speak at once, array-measured location separates speakers better than conventional spectral features.

### 2.3 Privacy-friendly acoustic social sensing

- Wyatt et al., [Inferring Colocation and Conversation Networks from Privacy-Sensitive Audio](https://pac.cs.cornell.edu/pubs/tist_colocation.pdf) (ACM TIST): infers colocation and conversation networks from privacy-preserving audio, studying face-to-face interaction among 24 graduate students over an academic year; the two kinds of network turned out to differ substantially. Their earlier work proposes features rich enough to capture conversational style but not sufficient to reconstruct intelligible speech, which can support the "no raw audio stored" narrative.

### 2.4 UWB and orientation

- [UWB tags for communication visualization and community structure estimation](https://arxiv.org/pdf/2510.06797): about 30+ participants, positioning error of 10 to 50 cm, with written informed consent.
- [Protractor](https://www.cs.columbia.edu/~xia/publication/ubicomp18-protractor/ubicomp18-protractor.pdf) (UbiComp 2018): wearable light tags that track interaction distance and relative body orientation. This addresses the limitation that UWB gives position but not orientation.
- [RFID face-to-face interaction study](https://arxiv.org/pdf/1406.6829): infers groups by transitivity (if A is in contact with B and B with C, then A and C are in the same interaction group). A simple post-processing idea we could borrow.

### 2.5 Low-power embedded acoustic localization (energy references)

- [ESP32 two-microphone DOA](https://e-archivo.uc3m.es/handle/10016/35679) (Fabregat et al., 2020): low-cost implementation with processing time under 3.3 ms, running in real time.
- [Ultra-low-power 4-microphone MEMS node](https://researchportal.be/en/node/291972): 1.4 mW when active and 66 nW asleep, transmitting only the angle rather than audio.
- [Owlet](https://www.cs.umd.edu/~nirupam/images/2_publication/papers/Owlet_MobiSys21_nirupam.pdf) (MobiSys 2021): low-power DOA using passive acoustic metamaterials, targeting the power and aperture costs of conventional arrays.

### 2.6 Surveys and adjacent work

- [F-formation detection survey](https://arxiv.org/pdf/2108.06181): covers methods, datasets and sensors, including the use of UWB and audio sensors; a good entry point.
- [MURMR](https://arxiv.org/pdf/2507.11797): multimodal group behavior analysis on MR headsets (already read; see the earlier discussion). Reusable ideas: sliding-window analysis, the list of turn-taking features, and the modality ablation design.

## 3. Implications for the Project

1. **Positioning of novelty:** acoustic plus DOA grouping has precedent. The value lies in modern methods, embedded deployment and energy, comparison with vision, and UWB fusion.
2. **The data strategy holds up:** I found no ready-made dataset with an array, moving people and multiple groups, so simulation-first with a few real validation sessions is reasonable.
3. **Evaluation can borrow layer by layer:** CHiME-6 or DiPCo for the acoustic front end; ConfLab, SALSA or MatchNMingle for the grouping algorithm; our own simulation and real sessions end to end.
4. **There is evidence for semantic content:** MCoRec results show content is strong for grouping, but it needs ASR, which is computationally expensive, so it fits as a stretch goal.
5. **The privacy narrative has precedent:** Wyatt et al. can be cited for not retaining reconstructable speech.

## 4. Suggested Reading Order

1. Kawaguchi 2010 (closest prior work)
2. The ConfLab dataset paper (data and annotation format reference)
3. The MCoRec dataset paper (task setup and metrics for simultaneous conversations)
4. Wyatt et al. (privacy-friendly feature design)
5. The F-formation survey (background)

## 5. Notes on Use

- Most entries are based on abstracts and snippets only; I have not read the full texts.
- For dataset access, licensing and exact size, defer to the official pages.
- The conclusion that no dataset meets all the conditions applies only to this search and does not mean none exists.
