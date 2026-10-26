# Feasibility Test Plan: Localizing a Speaker with Multiple Single Microphones (2D)

Last updated: 2026-10-25

## 1. Research Question

**Core question**: Using only a few separately placed single microphones (no camera, no UWB at inference time), can we estimate a speaker's 2D position in a room?

This feasibility test answers only this one question. The camera is used only during data collection to provide position labels. It is not used at inference time.

### Scope

- 2D coordinates (x, y) only, no height.
- A single speaker (me), walking around and talking inside a chosen area.
- One room (the living room).
- The only claim is "positions can be estimated within the same room." We do not claim cross-room generalization, because that needs data from many rooms and this experiment's data volume cannot support it.
- No multi-speaker, conversation grouping, or identity recognition.

## 2. Test Setup

### 2.1 Devices

| Device | Role | Notes |
|---|---|---|
| iPhone x 2, iPad x 1 | Three single-microphone nodes | Record with Voice Memos in lossless format. The three devices are different models, see Section 4 (Risks) |
| Aqara surveillance camera | Source of position ground truth | Strong wide-angle distortion and modest resolution. Exact model to be confirmed |
| Computer | Receives and saves the camera stream, then does offline processing | |
| Tape measure, painter's tape | Measure coordinates, mark reference points | |

### 2.2 Space and Coordinate System

- Area: the square open floor space in the living room (from the camera view, between the sofa, coffee table and the two doors). Dimensions to be measured.
- Pick one corner as the origin. Define the x-axis and y-axis along two adjacent walls. Unit: meters.
- Place the three recording devices around the perimeter of the area (for example at three corners or edge midpoints). **Their positions stay fixed for the entire experiment.** Measure their coordinates in the same coordinate system with a tape measure, to centimeter precision.
- The devices must not block the camera's view or sit on the walking path. Keep all three at a similar height.

### 2.3 Position Ground Truth (Camera)

- Mark a grid of reference points on the floor with painter's tape (about 4x4 is suggested, covering the corners and edges of the area), and measure each point's (x, y) with a tape measure.
- These points serve two purposes: (1) fit the mapping from pixels to floor coordinates; (2) keep a few points out of the fit and use them to validate the error of the camera-based position labels.
- On the side of the image nearest the lens, the feet must stay fully in frame, otherwise positions in that zone are lost. The desk in the lower right blocks the feet, so do not walk there.

## 3. Experimental Steps

### Step 0: Preparation and Calibration

1. Measure the area dimensions, the reference point coordinates and the coordinates of the three devices.
2. Camera calibration, choose one:
   - Use OpenCV's lens model. Take a dozen or more photos of a printed checkerboard at different positions in the frame, estimate the distortion parameters, undistort, then apply a homography.
   - Or skip explicit calibration. Use ten or more well-spread floor reference points and fit a more flexible mapping than a homography (polynomial or thin-plate spline). It is only reliable inside the area covered by the reference points.
3. Stand on several points measured with the tape, compare the position computed from video with the true position, and obtain the **label error**. The model error must be clearly larger than this for the conclusions to hold.
4. Phone settings (on every device):
   - Settings > Voice Memos > Audio Quality > Lossless. I have not verified where this setting is on iPad, please check yourself.
   - Do not use "Enhance Recording". It removes reverberation, and reverberation is exactly the kind of cue we want to keep.
   - Keep the microphone mode on Standard. Do not enable Voice Isolation.
   - Automatic gain cannot be turned off, so absolute loudness is not reliable.
   - Turn on Do Not Disturb, keep the devices plugged in, and leave enough free storage.

### Step 1: Recording the Video Stream and Tracking the Person

1. Recording:
   - The Aqara G5 Pro supports local RTSP. It must first be enabled in the Aqara app. The port is 8554, and the resolutions are 1520p, 1080p and 360p. According to a review, the app defaults to 360p, so set it to a high resolution manually. **Whether other models support RTSP is not verified.**
   - On the computer, receive the RTSP stream with ffmpeg or OBS and save it to a file. Keep the original stream and the timestamps. RTSP can have buffering and jitter, so read each frame's timestamp from the file and do not assume a constant frame rate.
2. Person tracking (position labels do not need to be real time. Offline processing after recording is fine and lets us rerun with a better model):
   - Use an off-the-shelf pose model (such as YOLO pose or MediaPipe) to detect the ankles in every frame and take the midpoint of the two feet as the ground position.
   - Convert it to room coordinates (x, y) with the mapping from Step 0.
   - Discard low-confidence frames (feet hidden by furniture or cut off by the frame edge). Optional: smooth the trajectory.
   - Only I should be in the frame, to avoid detecting other people.
3. Do a small test first: record a short video of me walking a few steps in the open area, run the pipeline, and see whether ankle detection is stable before deciding whether the resolution is enough.

### Step 2: Data Collection

1. Start-up check: confirm the three devices have not moved and the camera is recording.
2. Synchronization: **clap** in front of the camera, with the hands clearly visible in the frame and the clap audible on all three recording devices.
3. Walk around the chosen area while talking (reading a book aloud is steadier than improvising), with these requirements:
   - Cover the whole area, including edges and corners. Vary speed, and include pauses.
   - Vary facing direction and volume, so the model does not learn "this sentence."
   - Do not repeat content.
4. Clap again at the end, to estimate the clock drift of each device.
5. Duration: about 30 minutes in total, but **split into several sessions of 5 to 10 minutes**, with a clap at the start and at the end of each session. This keeps drift from accumulating, and if one segment goes wrong only that segment has to be redone.
6. Also record 2 to 3 minutes of silence (room ambient noise).
7. Suggestion: keep one independent session (ideally on a different day, or at least after repositioning the devices) just for the test set.

### Step 3: Alignment and Building the Dataset

1. **Audio-to-audio alignment**: cross-correlate the claps to put the three audio tracks on one timeline, and estimate and correct linear drift from the start and end claps. This step needs the highest precision.
2. **Audio-to-video alignment**: use the frame of the clap to determine the offset. The precision requirement is much looser: when standing still a few tens of milliseconds do not matter, but while walking at about 1 m/s, a 100 ms offset is roughly 10 cm.
3. **Slicing**: try window lengths of 1, 2 and 5 seconds, with a stride such as 0.5 seconds. The label of each window is the median position within the window (or the position at the window midpoint, choose one and keep it consistent).
4. **Filtering**: discard windows with too little speech (use voice activity detection) and windows with low position confidence. Windows with fast movement can be flagged separately as a distinct evaluation subset.
5. **Metadata**: record session ID, time, position, speed and speech ratio.
6. **Train / validation / test split**:
   - **Do not shuffle windows randomly.** Adjacent windows are nearly identical, so a random split leaks information and inflates accuracy.
   - Split by contiguous time blocks, leaving a gap of at least one window length between blocks.
   - Also do a **spatial hold-out**: an entire region never appears in training and only appears in testing. This best reflects real localization ability.
   - An independent session (different time) serves as the strictest test.

### Step 4: Models and Evaluation

**Baselines**

- Baseline 0: always predict the mean (or the room center) of the training positions. Every model must be clearly better than this.
- Baseline 1: classical time-difference-of-arrival. Compute GCC-PHAT for the three microphone pairs, then solve with multilateration. This needs known device coordinates and small enough synchronization error: with a speed of sound of about 343 m/s, **1 millisecond corresponds to roughly 34 cm of distance difference**.
- Baseline 2: use only per-channel loudness or band energy (similar to a signal-strength fingerprint) with kNN or regression. Note that automatic gain disturbs absolute loudness.

**Learned models**

- A: GCC-PHAT features of the three microphone pairs -> MLP or CNN, regress (x, y).
- B: log-mel spectrograms of the three channels, concatenated -> CNN, regress (x, y) or do grid classification (for example 0.5 m cells).
- C: fusion of A and B.

**Evaluation metrics**

- Mean and median position error (meters), 90th percentile error, error CDF.
- Fraction of predictions within 0.5 m and within 1 m.
- Error as a function of window length, speech ratio and location (center vs. edge).
- Learning curve: train with 25%, 50%, 75% and 100% of the training data and look at the test error, to judge whether the data is enough.
- Ablation: remove one of the three channels and see how much performance drops.

**Go/no-go criteria**: to be decided, to be filled in after discussing with Mani.

### Optional Extensions

- Move furniture and record another round. Test the model trained on the old data to quantify the effect of furniture changes on localization.
- Record stationary data on grid points as a cleaner supplementary evaluation.
- A second room, for a "simple room vs. complex room" comparison.

## 4. Known Risks and Mitigations

| Risk | Explanation | Mitigation |
|---|---|---|
| Synchronization across devices | Each of the three devices has its own clock, with different start times and drift. Time-difference-of-arrival methods are the most sensitive to this | Align with cross-correlated claps, estimate drift from the start and end claps; use simulation first to see how precise it needs to be |
| Different device models | Microphone sensitivity, frequency response and automatic gain differ, and the model may mistake them for position cues | Keep the device-to-position assignment unchanged for the whole experiment; record a calibration tone (such as a sweep or white noise) and compare the responses of each channel |
| Automatic gain | Voice Memos sets the recording level automatically and it cannot be adjusted manually | Do not rely on absolute loudness; third-party recording apps may offer manual gain, but I have not verified which one |
| Camera distortion and resolution | Wide-angle distortion, lower accuracy at the edges and far away | Calibration and validation in Step 0; do not treat far areas as the main area |
| Feet occluded | Furniture hides the ankles and the position is lost | Discard low-confidence frames; aim the camera downward as much as possible |
| Blurry labels | While walking, the position changes within a window | Use one consistent labeling rule; evaluate stationary and moving windows separately |
| Data leakage | Random window splits inflate accuracy | Split by time blocks and spatial regions |
| Data volume | 30 minutes covers only one arrangement of one room | Draw the learning curve; claim only "within the same room" |
| Camera model | Not every Aqara model supports RTSP | Confirm the model first |

## 5. Related Work and Datasets

Verification level: the content below comes from paper bodies or abstracts. I have not read each paper in full, and I note how much I actually read.

- **Liu et al., "Deep Learning Based Stage-wise Two-dimensional Speaker Localization with Large Ad-hoc Microphone Arrays"** (arXiv 2210.10265, full text read, not verified whether formally published)
  - Each node is a 4-microphone array. A CNN first estimates the direction, then node selection, triangulation and mean-shift clustering produce 2D coordinates. Trained on simulation only and tested on real data.
  - On real data (Libri-adhoc-nodes10, 10 nodes, loudspeaker playback), single-source position error is about 0.23 m (conference room) and 0.26 m (office).
  - Very sensitive to direction error: for a single source with circular arrays, a 1 degree direction error corresponds to about 0.06 m position error, 5 degrees to about 0.4 m, and 15 degrees to about 1.4 m.
  - Limitations: 6 to 10 nodes, and it did not test fewer than 6; node positions and orientations must be known; the source is a loudspeaker, not a real person.
- **Libri-adhoc-nodes10**: real recordings, 10 four-microphone linear-array nodes in each of two rooms, LibriSpeech played through a loudspeaker, with speaker positions, node positions and node rotation angles. Downloadable from GitHub. Whether the nodes are synchronized and the license terms are not verified.
- **Gong et al., "End-to-end Two-dimensional Sound Source Localization With Ad-hoc Microphone Arrays"** (arXiv 2210.08484, only the abstract and snippets seen): each node has a single microphone, and localization is formulated as classification. Closest to our setup. **The full text should be read next.**
- **Libri-adhoc40**: 40 strongly synchronized distributed nodes in a real office, loudspeaker playback. Whether position labels exist is not confirmed.
- **AV16.3**: real people, two 8-microphone circular arrays and 3 cameras, 3D mouth position error at most 1.2 cm. Only two compact arrays on a table, not distributed. Free, citation required.
- **LOCATA, RealMAN, STARSS22**: compact arrays, mainly for direction estimation. Not the same problem as our distributed single-microphone setup, for reference only.
- **DIRHA-English**: 32 synchronized microphones in an apartment, real people, with speaker position annotations. Paid access through LDC.

## 6. Open Items

- [ ] Exact model of the Aqara camera: can RTSP be enabled, can audio be obtained, can it be set to 1080p or higher
- [ ] Dimensions of the square area in the living room (length, width, in meters)
- [ ] Whether any furniture in the area must be moved or avoided
- [ ] Placement and coordinates of the three recording devices
- [ ] Reverberation time of the room (clap, record, and estimate the decay, as input for simulation)
- [ ] Where the lossless setting for Voice Memos is on iPad
- [ ] Spacing of the reference point grid (depends on the area size)
- [ ] Go/no-go thresholds (discuss with Mani)
- [ ] Read the full text of Gong et al. on single-microphone nodes

## 7. Background Notes for a New Session

- Claude has no memory across conversations. In a new session, please have it read this document first.
- This is a course project and should be done step by step. Do not do everything at once. After each step, wait for confirmation before continuing.
- Decisions already made: do not use Kawaguchi's method; 2D localization only; position ground truth comes from the Aqara camera plus a vision model; the venue is the living room; the recording devices are two iPhones and one iPad; do simulation and baselines first, then real recordings.
- Discussed but not yet done: single-microphone simulation (pyroomacoustics, build a room matching the living room's dimensions, place 3 single-microphone nodes, test with classical methods, and sweep synchronization error, reverberation and noise); a reality check with Libri-adhoc-nodes10.
- The verification level of numbers and product information is marked in the text. Check anything not marked as verified before relying on it.