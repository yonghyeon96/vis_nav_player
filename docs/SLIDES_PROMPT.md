# 슬라이드 요청문 (ChatGPT용)

## 사용 방법
1. ChatGPT에 아래 **"요청문"** 블록 전체를 붙여 넣습니다.
2. 같은 대화에 다음 이미지 4장을 첨부합니다. 모두 이 `docs/` 폴더에 있습니다.

| 파일 | 내용 | 쓰는 슬라이드 |
|---|---|---|
| `goal_photos_midterm.png` | Midterm 목표 사진 4장(앞·왼·뒤·오른쪽, 1280×240 한 장) | 1 (문제) |
| `fig_perception.png` | 왼쪽: 카메라 화면의 바닥–벽 경계(빨간 점). 오른쪽: 그 점들을 위에서 본 모습과 0.4 m 벽 격자 | 2 (방법 1) |
| `fig_midterm_map.png` | 탐색 데이터로 만든 Midterm 지도, 1·2회차 경로, 사진으로 찾은 통로(초록) | 2 또는 3 |
| `doorway_12-4_11-4.png` | 목표 앞 통로가 열려 있다는 증거(윗줄은 1회차 주행 화면, 아랫줄은 탐색 화면) | 3 (방법 2), 작게 |

3. 필요하면 배경 자료로 `REPORT_NOTES_KO.md`도 첨부합니다. 다만 요청문만으로 충분하도록 썼습니다.
4. 결과를 받으면 숫자와 용어가 아래 "사실"과 맞는지 확인합니다.
   - ChatGPT가 없는 결과를 지어내거나, 우리가 쓰지 않은 방법(SLAM, 딥러닝 등)을 넣지 않았는지 봅니다.

`doorway_7-2_7-3.png`도 있지만 슬라이드가 3장뿐이라 넣지 않는 것을 추천합니다.

---

## 요청문 (여기부터 복사)i

You are helping a student team make **3 presentation slides** (16:9, English) for a course report. Please produce the slide content (titles, bullet text, figure placement, a diagram, speaker notes) and, if you can, a downloadable .pptx. I attached 4 images; use them where indicated. **Use only the facts below; do not invent results, methods or numbers.** Keep the slides clean and sparse: short bullets, few words, large figures.

### The grading rubric (the slides must answer exactly this)

- **Slide 1: Description of the Problem and References (2 pts).**
  - In a few sentences, describe the problem we are trying to solve. Only this one slide may be used for the answer.
  - Cite the most important papers, tools or libraries used in the solution.
- **Slides 2–3: Method (3 pts).**
  - Explain our strategy for navigation, using sentences, graphs or whatever conveys the ideas best.
  - At most 2 slides.

### Facts: the problem

- Course: NYU ROB-GY 6203 Robot Perception, Visual Navigation Challenge (AI4CE Lab), Midterm challenge.
- **Setting:**
  - A robot is in an unknown maze in a simulator that runs on the course server.
  - The maze is 15 × 15 cells of 0.4 m (about 6 m × 6 m), with corridors, some open rooms, and textured walls.
  - Each step the agent receives only a first-person RGB camera frame (320 × 240). It gets no position, no map and no depth.
- **Given before the run:**
  - **Exploration data:** frames recorded by an explorer robot driving through the maze, with its action labels. On the Midterm this was 3 drives and about 16,000 frames.
  - **4 photos of the goal:** front, left, back and right views from the goal pose.
- **The task:**
  - Choose discrete actions (FORWARD, BACKWARD, LEFT, RIGHT and their combinations, repeated for some ticks).
  - Drive to the goal, then call CHECKIN.
- **Scoring:** distance to the goal (≤ 0.1 m for full marks), heading error (≤ 10°), and **SPL**, i.e. the path length compared with the oracle shortest path in steps (≥ 0.95 for full marks). Missing the goal scores 0.
- **Hard parts:**
  - Every run has its own unknown motion bias: speed scale, turn scale, and a curvature while driving. There is also per-tick noise, so dead reckoning drifts.
  - The maze is new, and there are only 3 attempts.
  - Wall textures repeat, so image matching alone can confuse places.
  - The explorer did not necessarily drive every corridor.

### Facts: our method (pipeline)

The design idea: **the maze is a grid, so walls lie on the lines x, y = 0.2 + 0.4·k m.** Instead of learning, we use this geometric structure plus classical vision.

**A. Perception and localisation (each frame).**

- Segment the floor by its colour.
- Find the floor/wall boundary in each image column.
- Turn each boundary pixel into a point on the ground with flat-floor geometry: X = f·h / (v − c_y), with camera height h = 0.214 m and focal length f = 92 px.
- Fit these "wall feet" to the wall lines of the grid (x, y = 0.2 + 0.4k) by a weighted least-squares pose fit, with the motion prediction as the prior (a MAP fit).
- Result: an absolute pose (x, y, heading) in the maze frame every frame, at about cm accuracy, with no drift.
- Figure: `fig_perception.png`.

**B. Map building from the exploration data (offline, before connecting).**

- Run the same lattice registration on every exploration frame to recover the explorer's path in cell coordinates.
  - Rotation between frames is measured from the image with pyramidal Lucas–Kanade optical flow on Shi–Tomasi corners.
  - The capture interval is estimated automatically.
- Drives that do not start at the spawn are placed by their overlap with the drives already placed.
- The map is the set of cells and the connections the explorer drove.
- **Map cleanup:**
  - A connection is removed when the photos show a wall across the doorway.
  - **A connection is added when frames taken near a cell centre, looking straight at a never-driven doorway, see floor running through it.** This was the key Midterm improvement; see the results below.
- **Goal localisation:**
  - The 4 goal photos are matched with SIFT against every exploration frame.
  - Each photo votes once per place, and places seen by more of the photos rank higher.
  - On the Midterm the top candidate had a support of 104 against 20 for the second.
- Figure: `fig_midterm_map.png`.

**C. Planning.**

- Dijkstra on (cell, heading) states.
- The cost is about the number of steps:
  - a cell costs 0.4 m / 0.03 m ≈ 13 ticks;
  - a connection driven only once, or only seen, costs 3×;
  - a corner costs about 1.5 cells.
- So among equally long routes, the one with fewer corners wins.

**D. Driving (a cell-by-cell controller).**

- Follow the waypoints at cell centres.
- Rotate in place at corners. Only the last 10° before a corner is turned while moving: posts leave only about 3 cm of lateral slack.
- Correct heading with forward+turn arcs.
- Learn the run's speed scale, turn scale and curvature online from the corrections that localisation makes.
- If a planned doorway shows a wall (seen in 2 frames while facing it), delete that connection and replan.

**E. Final approach (pose relative to the goal from the goal photos).**

- Within 0.35 m of the goal, pair SIFT features between the live frame and the goal photos.
- Each pair gives the same wall point in both views, through the floor boundary. The goal photos were taken at a camera height of 0.163 m; a height check rejects wrong pairs.
- A RANSAC 2-D rigid fit gives the robot pose in the goal frame.
- The fit is accepted only if at least 70% of the camera's wall points agree with the 360° wall outline built from the 4 goal photos.
- Locally the error is 4 mm / 0.3° (median).
- CHECKIN once the robot is within 5 cm and 3° of the goal.
- If the place does not look like the goal, try the next candidate.

**Results (for a small footer or speaker notes only; the rubric does not ask for results):**

- Development maze: 9–10/10 in most runs, arriving within 2–4 cm.
- **Midterm, attempt 1: 9/10** (997 steps, SPL 0.88, 1.4 cm / 1.6°).
  - From SPL = oracle / ours, the oracle was about 878 steps, but no route on our map could be shorter than about 1011.
  - The cause was a doorway next to the goal that the explorer had looked through but never driven (see `doorway_12-4_11-4.png`). We added such "seen-open" doorways (B) and a corner cost (C).
- **Midterm, attempt 2: 10/10** (852 steps, SPL 1.00, 2.7 cm / 0.6°).
- Only 2 of the 3 attempts were used.

### Facts: references to cite (slide 1, short form is fine)

- NYU AI4CE Lab, Visual Navigation Challenge starter kit and SDK (`vis_nav_player`, `vis_nav_sdk`, `vis-nav-sim`).
- OpenCV (Bradski, 2000) and NumPy (Harris et al., Nature 2020).
- SIFT: D. G. Lowe, "Distinctive Image Features from Scale-Invariant Keypoints," IJCV 2004.
- RANSAC: M. A. Fischler and R. C. Bolles, CACM 1981.
- Optical flow: B. D. Lucas and T. Kanade, IJCAI 1981 (pyramidal implementation by J.-Y. Bouguet, 2000). Corners: J. Shi and C. Tomasi, "Good Features to Track," CVPR 1994.
- Least-squares rigid alignment: S. Umeyama, IEEE TPAMI 1991.
- Shortest paths: E. W. Dijkstra, 1959.
- SPL metric: P. Anderson et al., "On Evaluation of Embodied Navigation Agents," arXiv:1807.06757, 2018.
- Optional, as the kit baseline we studied first but did not use: RootSIFT (Arandjelović and Zisserman, CVPR 2012) and VLAD (Jégou et al., CVPR 2010).

### What I want for each slide

**Slide 1: "Problem & References"**

- 3–4 sentences on the problem: unknown maze, camera only, exploration data plus 4 goal photos, CHECKIN at the goal, scored on distance, heading and SPL, unknown per-run motion bias.
- Include `goal_photos_midterm.png` as a thin strip labelled "the 4 goal photos (input)".
- A compact reference list in small font, grouped as "Tools" and "Methods".

**Slide 2: "Method (1/2): a map and a pose from wall geometry"**

- Left: `fig_perception.png` with the caption "floor/wall boundary → ground points → fit to the 0.4 m wall grid → absolute pose every frame".
- Right: `fig_midterm_map.png` with the caption "explorer path re-localised the same way → cell map; green = doorways added from photos; goal = SIFT votes of the 4 goal photos".
- 3–4 short bullets covering A and B.

**Slide 3: "Method (2/2): plan, drive, approach"**

- A horizontal flow diagram (draw it as shapes): **Exploration data → Cell map + goal candidates → Dijkstra (steps + corner cost) → Cell-by-cell driving with lattice localisation (learns motion bias; replans on blocked doorway) → Goal-photo pose approach (SIFT + RANSAC) → CHECKIN**.
  - Add a dashed fallback arrow: "not the goal → next candidate".
- 3–4 short bullets covering C, D and E.
- A small inset of `doorway_12-4_11-4.png`, cropped to one top-row frame, with a one-line results footer:
  "Midterm: attempt 1 9/10 (SPL 0.88) → found a never-driven doorway → attempt 2 10/10 (852 steps, SPL 1.00)".

**Speaker notes:** 3–5 sentences per slide, in plain language.

## 요청문 끝

---

## 참고: 슬라이드 후보 사진 더 보기

- **2·3회차 실행 기록 프레임:** `artifacts/runs/<시각>_da90a443e215/frames/`
  - 저장소에는 올라가지 않습니다.
  - 2회차는 `20261029-031025_da90a443e215`이고, 마지막 프레임 근처가 목표 앞 장면입니다.
- **다시 만들기:**
  - `fig_perception.png`와 `fig_midterm_map.png`는 2회차 기록과 Midterm 데이터로 만들었습니다.
  - 다른 프레임으로 다시 만들려면 Claude에게 "다른 프레임으로 fig_perception 다시"라고 요청하면 됩니다.
