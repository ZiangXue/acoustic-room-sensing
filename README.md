# Project Proposal (Draft): Non-Visual, Low-Power Sensing of Social Interaction

Joseph Seok, Ziang Xue

## Question

How well can low-power, non-visual sensing (microphone arrays, optionally UWB) detect social interaction within a room, **running entirely on-device with no data sent to any server**, and how does it compare to a vision-based approach in accuracy and energy cost?

## Scope

We will design, train, and deploy an AI model on an embedded device that takes multi-channel acoustic signals from a microphone array and, in a single indoor room with up to ~10 people (primary evaluation at ~4-6), 

1. detects conversation subgroups (typically 2-3), and
2. assigns each person a **persistent anonymous identifier**, so the same person can be tracked across a session even as they move or switch groups.

All inference runs on the device itself; no raw data is sent to a cloud or data center. Identifiers do not correspond to real identities and are only valid within a single session. The model architecture is part of the project and is not fixed in advance.

**Stretch goal:** Add a UWB tag per person as an additional input, and measure whether and how much it improves subgroup detection over acoustic-only.

**Known limitation:** Acoustic sensing only detects people who speak. Silent individuals cannot be assigned to a group, so evaluation is defined over active speakers.

## Methodology (high level)

1. **Simulation.** Generate multi-channel acoustic data (and ground-truth groups, IDs, positions) with pyroomacoustics, driven by a simulated behavior model of group formation, switching, and turn-taking. Train and explore model architectures on a PC. Zephyr's simulation targets can be used for early on-device pipeline testing before physical deployment.
2. **On-device deployment.** Deploy the trained model to an embedded platform with a microphone array, and measure latency, memory, and energy.
3. **Real-world validation.** Deploy in a lab / office-hour setting with consent, and compare device output against human-annotated ground truth (with video as a reference).

## Evaluation

**Form.** Two axes, compared against a vision-based baseline to characterize the accuracy-energy tradeoff:

- **Accuracy:** how reliably the system recovers the correct subgroups (e.g., "persons A, B, C are in the same group"), and keeps anonymous IDs consistent over time.
- **Energy / resource cost:** power, latency, and memory on the embedded device.

**Ground truth.**

- *Simulation:* known by construction (group membership, IDs, positions, speech activity).
- *Real-world:* human annotation of group membership over time, made by annotators (not participants) reviewing video.

**Experiments.**

1. *Simulation:* vary conditions (number of people/groups, speech overlap, reverberation, noise) to see how accuracy degrades and where the system fails.
2. *Real-world deployment:* run the device live in the target setting and compare its output to human-annotated ground truth. This is the final evaluation.
3. *Energy:* measured on the real device, since simulation cannot give true power numbers.

**Reporting.** Simulation and real-world results are reported separately; the gap between them is itself a result. Specific metrics will be chosen later.

## Constraints

- Real data will be limited, hence the simulation-heavy development and a small number of real sessions.
- Real recordings require appropriate consent / approval.

## Questions for Mani

1. **Consent / IRB:** We plan to record multi-channel audio (and possibly video) of classmates in a lab setting. Do we need IRB approval or written consent, and what is the process?
2. **Hardware:** Is the bar for "embedded" MCU-level, Pi-level, or either?
3. **Scope boundary:** What level of completion would you consider solid for the core project? Is UWB acceptable as a stretch goal?
4. **Baseline:** Do we need a vision baseline, and should we implement it ourselves or use an off-the-shelf model and measure its energy?
5. **Simulation:** We plan to use pyroomacoustics. Any other tools you'd suggest? Is a simulation-heavy plan with a small number of real sessions for final validation reasonable?
6. **Real evaluation scale:** Are one or two real sessions sufficient as the final evaluation?