# Changelog

All notable changes to Tessera will be documented in this file.

## [Unreleased]

### Planned
- Phase III: Connected Components (連の完全実装)
- Phase III: 地の計算
- External engine evaluation (GnuGo/KataGo)

---

## [0.4.1] - 2026-01-13

### Fixed
- **Critical: MambaStateCapture memory leak**
  - Disabled `MambaStateCapture` in `ParallelTrainer`
  - Root cause of OOM after 3 ELO evaluations
  - Single line fix enabled 100,000 game completion

### Added
- VRAM monitoring in logs (debug mode)

---

## [0.4.0] - 2026-01-13

### Added
- **Clean Room Protocol** for ELO evaluation
  - `VRAMSanitizer` context manager
  - Automatic VRAM cleanup on enter/exit
- **CPU Staging** for checkpoint loading
  - `map_location='cpu'` prevents VRAM pollution
  - Selective transfer of weights only
- **Caller Responsibility** pattern
  - Training loop releases resources before ELO
  - Engine rebuild after evaluation
- **ACD Checkpoint Saving**
  - Atomicity: temp file + rename
  - Consistency: post-save verification
  - Durability: fsync to disk

### Changed
- `elo.py` upgraded to v1.3.0
- `long_training_v4.py` upgraded to v4.1.0
- ELO batch size reduced to 64 (from 32)

---

## [0.3.0] - 2026-01-13

### Added
- **ELO Rating System** (`elo.py` v1.0.0)
  - Bradley-Terry model
  - Match against past checkpoints
  - JSON Lines logging
  - Tile-based ELO tracking
- **Parallel Training** (`long_training_v4.py`)
  - BATCH_SIZE = simultaneous games
  - Individual game reset on completion
  - Graceful shutdown with SIGINT/SIGTERM
- **Discord Monitoring** (`monitor.sh`)
  - Process watchdog (60s interval)
  - Webhook notifications on failure

### Changed
- Training architecture from serial to parallel
- Game completion triggers immediate learning

---

## [0.2.2] - 2026-01-12

### Added
- `GPUGoEngine.reset_selected()` for partial reset
- Temperature scheduling in training
- Auto-scaling based on VRAM

### Fixed
- SSM divergence with norm limit (300.0)
- Gradient clipping (0.5)

---

## [0.2.0] - 2026-01-12

### Added
- **GPU Native Go Engine** (`gpu_go_engine.py`)
  - Batched board state (batch, 2, 19, 19)
  - Vectorized operations (no Python loops)
  - Single stone capture
  - Simple ko detection
- **Mamba Model** (`model.py`)
  - State Space Model for move prediction
  - 4-layer architecture (d_model=256)
- **Monitoring** (`monitor.py`)
  - VRAM tracking
  - SSM state norm monitoring
  - GPU temperature alerts

---

## [0.1.0] - 2026-01-11

### Added
- Initial project structure
- Docker environment with CUDA support
- Basic training loop

---

## Version Numbering

- **Major (X.0.0)**: Phase completion
- **Minor (0.X.0)**: Feature additions
- **Patch (0.0.X)**: Bug fixes

---

## Links

- [Design Spec Phase II](docs/DESIGN_SPEC_PHASE_II.md)
- [Experiment Log 2026-01-13](docs/EXPERIMENT_LOG_20260113.md)
- [Architecture Note: GPU-Native Sovereignty](docs/ARCHITECTURE_GPU_NATIVE.md)
