# fusion_tracking API reference {#mainpage}

`fusion_tracking` tracks an unknown number of vehicles from lidar and radar
detections. fusion::MultiTargetTracker takes one sensor frame at a time;
everything else is a component it is built from, or the simulation and
evaluation around it.

| Part | Entry point | Implementations |
|---|---|---|
| Motion and measurement models | fusion::ctrvPredict(), fusion::measure() | CTRV; lidar position, radar range / bearing / range rate |
| State estimation | fusion::Filter, fusion::makeFilter() | fusion::Imm, fusion::Ukf, fusion::Ekf, fusion::NaiveUkf |
| Association | fusion::MultiTargetTracker, fusion::solveAssignment() | global nearest neighbour with χ² gating; oracle labels |
| Track management | fusion::ScoreParams | log-likelihood-ratio score; hit count |
| Simulation | fusion::Simulator, fusion::LidarScanner | moving ego vehicle; point detections, or ray-cast scans with box fitting |
| Evaluation | fusion::Evaluator, fusion::runSimulation() | RMSE, NEES, NIS, OSPA, ID switches |
| Configuration | fusion::Config, fusion::loadConfig() | YAML with includes; unknown keys rejected |

Minimal use with your own detections:

```cpp
#include "fusion/config.h"
#include "fusion/tracking/tracker.h"

fusion::MultiTargetTracker tracker(fusion::loadConfig("config/default.yaml"));
// For each sensor frame, in time order:
tracker.process(time, fusion::Sensor::Radar, detections, ego_pose);
for (const fusion::Track& t : tracker.tracks())
    if (t.confirmed) { /* t.id, t.filter->state(), t.filter->covariance() */ }
```

Design notes, trade-offs and measured results are in `docs/DESIGN.md` in the
repository.
