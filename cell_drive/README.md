# cell_drive — 칸 단위 주행기 (기술 참고)

`source/my_agent.py`가 이 패키지의 `CellAgent`를 그대로 쓴다. 개발 과정과 실패 기록은 [../REPORT_NOTES_KO.md](../REPORT_NOTES_KO.md)에 있다.

이 주행기가 의지하는 사실은 하나다. **미로는 0.4 m 격자이고, 모든 벽은 격자선 위에 있다.** 주행기 좌표계는 출발 칸 중심이 원점이고, 처음 바라보는 방향이 +x다. 벽은 x, y = 0.2 + 0.4k 위에 있다.

## 처리 흐름

```
탐색 데이터 ──► explore_map  탐색자 경로를 칸 좌표로 복원 → 칸 연결 지도
목표 사진  ──► goal         탐색 사진과 대조 → 목표 후보 위치·방향 (순위)
                navigator    경로 계획 → driver 주행 → goal_pose 정밀 접근 → CHECKIN
카메라 영상 ──► perception   벽 바닥점 → 격자 정합으로 칸 안의 위치·방향
```

| 파일 | 역할 |
|---|---|
| `perception.py` | 바닥색으로 열마다 바닥–벽 경계를 찾아 바닥 좌표로 바꾼다(카메라 높이 0.214 m, fx = fy = 92). 6열 폭의 경계 방향으로 각 점의 벽 방향을 정하고, 점들이 격자선에 오도록 위치와 방향을 맞춘다(MAP: 움직임 예측을 사전분포로 사용). |
| `driver.py` | 칸 중심 경로점을 따라간다. 회전은 칸 중심에서 제자리로 하고, 직진은 최대 3 tick 단위다. 방향 오차는 `FORWARD\|LEFT/RIGHT` 아크로 고친다. 코너에 들어가기 직전 마지막 10°는 이동하며 돈다. 실행마다 다른 속도·회전 배율과 곡률은 보정량에서 학습한다. 통로를 정면으로 볼 때 두 프레임 연속으로 벽이 보이면 막힌 통로로 판정한다. |
| `explore_map.py` | 탐색 사진마다 같은 격자 정합으로 탐색자 위치를 복원한다. 회전량은 영상(LK)에서 재고, 라벨로는 이동 여부만 본다. 사진 간격(tick)은 자동 추정한다(로컬 5, Development 3, Midterm 1). 출발점에서 시작하지 않는 궤적은 겹치는 장면으로 위치를 맞춘다(`align`). 탐색 사진에 문 가운데 벽이 보이는 연결은 지운다(`walled_edges`). 지나가지 않았지만 칸 중심에서 정면으로 본 사진들에 바닥이 경계 너머로 이어지는 연결은 더한다(`seen_open_edges`). 경로는 Dijkstra로 찾는다(한 번만 지나간 연결과 본 연결은 비용 3배, 코너 하나는 칸 약 1.5개). |
| `goal.py` | 목표 사진 4장(앞·왼·뒤·오른쪽)을 모든 탐색 사진과 SIFT로 대조한다. 목표 사진의 카메라 높이는 0.163 m다. 후보 장소는 사진 하나당 한 표, 장소를 본 사진 수에 비례한 점수로 고른다. |
| `goal_pose.py` | 목표 사진만으로 로봇의 목표 기준 자세를 구한다. SIFT 짝마다 두 영상의 바닥 경계로 벽 위치를, 행으로 높이를 구해 같은 점인지 확인한다. 그 뒤 RANSAC 강체 변환을 구하고, 목표 둘레 360° 벽 윤곽과 70% 이상 겹쳐야 받아들인다. |
| `navigator.py` | 단계 관리: `plan → drive → approach → checkin`. 마지막 구간에서 목표가 0.35 m 안에 들어오면 `goal_pose` 조향으로 넘긴다. 목표 5 cm·방향 3° 안이면 CHECKIN한다. 목표 사진으로 위치를 잴 수 없으면 예전 확인 단계(`verify/servo`)로 넘어가고, 실패하면 다음 후보로 간다. 경로의 코너 비용은 `TURN_WEIGHT`(0.76)다. |
| `agent.py` | SDK `Agent`(`CellAgent`). 지도와 후보 계산은 모두 `__init__`에서 끝낸다(서버 연결 전). 지도 캐시는 `artifacts/cell_drive/cache/`에 둔다. 실제 실행마다 `artifacts/runs/<시각>_<챌린지>/`에 `log.txt`, `trace.tsv`, 결정마다의 카메라 사진을 남긴다. |
| `run_route.py` | 로컬 주행 시험. 정답 최단 경로를 주고, 평가기만 정답을 보며 매 tick 벽 접촉을 잰다. |
| `run_goal.py` | 로컬 전체 시험: 탐색 데이터 → 목표 → CHECKIN. 서버 예산 10000, 편향 on. |

## 실행

```powershell
uv run source/my_agent.py                    # .env의 challenge (Development)
uv run source/my_agent.py --local 12         # 로컬 미로 12 (첫 실행 때 탐색 데이터 녹화)

uv run python -m cell_drive.run_route --seeds 7 8 9 10 11 12 13 14 15        # 주행 시험
uv run python -m cell_drive.run_route --seeds 7 --viewer                      # 눈으로 보기
uv run python -m cell_drive.run_goal --maze 12 --data artifacts/cell_drive/explore/local-12
```

- `run_goal`의 탐색 데이터(`artifacts/cell_drive/explore/local-9..12`)는 서버와 같은 3 tick 간격으로 녹화한 것이다. 없으면 `vis_nav_sim.Simulator.record`로 다시 만들 수 있다.
- 결과는 `artifacts/cell_drive/`에 저장한다. 이 폴더는 Git에서 제외된다.

## 현재 성능 (2026-10-28)

| 시험 | 결과 |
|---|---|
| 로컬 주행(정답 경로, 편향 on), 미로 7–30 + 강한 편향 | 39/41 완주, 완주한 경로에서 벽 접촉 0 |
| 로컬 전체(미로 9, 10, 12) | 모두 perfect, 목표 1–4 cm |
| Development 서버(정밀 접근 추가 후 9회) | 9회 모두 목표 2.6 cm·1.7° 안에 도착. 10점 3회, 9점 5회, 6점 1회(이후 고친 오판) |

## 측정한 시뮬레이터 수치

| 항목 | 값 |
|---|---|
| 이동 | 0.03 m/tick 전진, 3.4377°/tick 회전. `FORWARD\|LEFT`는 0.0299 m + 3.437°/tick |
| 카메라 | 높이 0.214 m (목표 사진은 0.163 m), fx = fy = 92, cx = 160, cy = 120 |
| 몸체·접촉 | 벽 정면 정지 시 중심–벽 0.0595 m. 벽에 붙은 채 회전하면 89° 명령이 약 34°만 돈다 |
| 기둥 | 중심에서 약 0.17 m에서 걸린다. 통로 중앙선 옆 여유는 약 3 cm |
| 서버 편향 | 실행마다 속도·회전 배율, 직진 곡률 0.06–0.8 rad/m. SDK `LocalSession`은 편향을 넣지 않는다 |
| oracle step | 이동 거리 / 0.03 + 회전각 / 3.4377 (제자리 회전을 포함한 0.2 m 격자 최단 경로) |
