Project Information

Title of the Project

A working title is fine - you can refine it later

Chatter - Deteching social interaction in a room with non-visual modalities

Research question or design

What specific question will you investigate, or what capability will you build and evaluate? Identify the physical/sensing setting and the relevant systems constraint. “Explore LLMs for sensing” is too broad. [~60 words]

> Using non-visual sensing modality, including microphone arrays, UWB tags, and/or thermal to design a device and model to detect social interaction within a room. Specifically, we'd like to detect sub-group dynamics like discussion and conversation among individuals in a single room, while maintaining tracking for each individual. Examine how these modality compare to a vision-based approach in accuracy and energy cost.

Motivation and starting point

Why is this worth investigating? Identify 1–3 relevant papers, systems, or repositories and explain what you will build on or examine. Distinguish your proposed course work from existing or concurrent research. [~80 words]

Approach and scope 

Describe your proposed approach and minimum end-of-quarter deliverable. State what is outside the required scope; identify at most one stretch goal. [~100 words]

We will train a model on simulated data (pyroomacoustics) and deploy it on a Raspberry Pi with a ReSpeaker microphone array and a UWB receiver, using tags worn by participants for distance/direction sensing. The model fuses acoustic and UWB signals to detect conversation subgroups and assign each person an anonymous ID. The device will also record real sessions, which we film and human-annotate as ground truth.
Minimum deliverable: a working pipeline and device for one room (≤10 people, 2-3 groups), evaluated in simulation and in at least one real session, with measured accuracy and energy. Out of scope: real-identity recognition, multi-room settings, detecting silent invididuals. Stretch goal: adding speech content (semantics) to measure its effect on group detection.

Feasibility, resources, and fallback

@Joseph What code, data, hardware, compute, or API access do you need? What is already available versus uncertain? Identify the largest risk and a simpler fallback that still makes it a viable project ad answers a useful question. [~80 words]

Next steps and responsibilities

@Joseph What concrete component or key assumption will you test by the next milestone in two weeks? Give an initial division of responsibilities among team members. [~60 words]

URL to your project repository at GitHub

https://github.com/ZiangXue/chatter

URL to your GitHub pages project website

https://github.com/ZiangXue/chatter
