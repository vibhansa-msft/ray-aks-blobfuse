# train_minimal.py - Workflow & Architecture Diagram

## System Overview Block Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DISTRIBUTED TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────────┘

                                 HEAD NODE
                        ┌──────────────────────┐
                        │   Ray Cluster Init   │
                        │   (Master Process)   │
                        └──────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
         ┌──────────▼──────┐  ┌──▼──────────┐  │
         │ Model Caching   │  │ Checkpoint  │  │
         │ (HuggingFace)   │  │ Dir Setup   │  │
         └─────────────────┘  └─────────────┘  │
                                                │
                    ┌───────────────────────────┘
                    │
         ┌──────────▼──────────────┐
         │ Parquet File Discovery  │
         │ (glob PARQUET_PATH)     │
         └────────────┬────────────┘
                      │
         ┌────────────▼─────────────┐
         │  Launch TorchTrainer     │
         │  (NUM_WORKERS=20)        │
         └────────────┬─────────────┘
                      │
        ┌─────────────┼─────────────┐
        │ DISTRIBUTED ACROSS NODES  │
        │ (DistributedDataParallel) │
        │                           │
    ┌───▼──────┐ ┌───▼──────┐ ┌───▼──────┐
    │ Worker 0 │ │ Worker 1 │ │ Worker N │
    └──────────┘ └──────────┘ └──────────┘
```

---

## Detailed Component Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            WORKER EXECUTION FLOW                            │
│                        (Runs on Each Ray Worker)                            │
└─────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────────┐
    │ INITIALIZATION                                                      │
    │ ├─ Get rank (worker ID) and world_size (total workers)             │
    │ ├─ Start overall timer                                             │
    │ └─ Receive config from head node (parquet_files list, dirs)        │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ DATA LOADING                                                        │
    │ ├─ Randomly select ONE parquet file from available list            │
    │ ├─ Read parquet file into pandas DataFrame                         │
    │ ├─ Record: parquet_load_time, parquet_size_mb, row count          │
    │ └─ Randomly select ONE row from DataFrame                          │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ MODEL INITIALIZATION                                               │
    │ ├─ Load pretrained model from local cache                          │
    │ ├─ Move model to device (CPU)                                      │
    │ ├─ Record: model_load_time, model_size_mb                          │
    │ └─ Create optimizer (AdamW)                                        │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ TOKENIZER INITIALIZATION                                           │
    │ ├─ Load tokenizer from local cache                                 │
    │ ├─ Set padding token to EOS token                                  │
    │ └─ Record: tokenizer_load_time                                     │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ TRAINING SIMULATION                                                │
    │ ├─ Simulate training for 10 seconds (time.sleep)                   │
    │ ├─ Generate mock loss value: 2.5 + (rank * 0.1)                   │
    │ └─ Record: training_simulation_time                                │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ WORKER SYNCHRONIZATION                                             │
    │ ├─ Barrier: All workers wait here                                  │
    │ └─ Ensures all workers complete before checkpoint save             │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ CHECKPOINT SAVING                                                  │
    │ ├─ Get model state dict                                            │
    │ ├─ Create checkpoint data dict (state + metadata)                  │
    │ ├─ Save to: worker_{rank:03d}_checkpoint.pt                        │
    │ ├─ Record: checkpoint_save_time, checkpoint_size_mb                │
    │ └─ Record: overall_time (end-to-end worker time)                   │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
    ┌────────────────────────▼────────────────────────────────────────────┐
    │ METRICS REPORTING                                                  │
    │ ├─ Compile all metrics into dictionary:                            │
    │ │  - loss, rank, world_size                                        │
    │ │  - parquet_rows, parquet_load_time, parquet_size_mb              │
    │ │  - model_load_time, model_size_mb                                │
    │ │  - tokenizer_load_time                                           │
    │ │  - training_simulation_time                                      │
    │ │  - checkpoint_save_time, checkpoint_size_mb                      │
    │ │  - overall_time                                                  │
    │ └─ Report via ray.train.report(metrics)                            │
    └────────────────────────┬────────────────────────────────────────────┘
                             │
                    WORKER EXECUTION COMPLETE
```

---

## Complete End-to-End Workflow (Sequence Diagram)

```
HEAD NODE                           WORKERS (0...N)
┌─────────┐                        ┌──────────────┐
│         │                        │              │
├─ Init Ray                        │              │
│         │                        │              │
├─ Cache Model ────────────────────► Shared Dir:  │
│  (HuggingFace)                   │  base_models/│
│         │                        │              │
├─ Setup Checkpoint Dir            │              │
│         │                        │              │
├─ Discover Parquet Files          │              │
│  glob(PARQUET_PATH)              │              │
│         │                        │              │
├─ Launch TorchTrainer             │              │
│  with NUM_WORKERS config         │              │
│         │                        │              │
├─ Send config (files list)────────┤ [Worker 0]   │
│         │    ├─ Select random file
│         │    ├─ Load parquet (1 row)
│         │    ├─ Load model
│         │    ├─ Load tokenizer
│         │    ├─ Train sim (10s)
│         │    └─ Save checkpoint
│         │                        │              │
├─ Send config (files list)────────┤ [Worker 1]   │
│         │    ├─ Select random file
│         │    ├─ Load parquet (1 row)
│         │    ├─ Load model
│         │    ├─ Load tokenizer
│         │    ├─ Train sim (10s)
│         │    └─ Save checkpoint
│         │                        │              │
│         │    ┌──────────────────────────────┐   │
│         │    │  All workers sync (barrier)  │   │
│         │    └──────────────────────────────┘   │
│         │                        │              │
│ ◄──────────── Collect metrics ◄──┤ Report      │
│   [All Workers]                  │  metrics    │
│         │                        │              │
├─ Aggregate statistics            │              │
│  (timing, sizes, row counts)     │              │
│         │                        │              │
├─ Print worker summary            │              │
│  (per-worker & aggregate)        │              │
│         │                        │              │
├─ Consolidate checkpoints         │              │
│  (average all worker states)     │              │
│         │                        │              │
├─ Save final model                │              │
│         │                        │              │
└─ Shutdown Ray                    │              │
                                   └──────────────┘
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATA & STORAGE FLOW                             │
└─────────────────────────────────────────────────────────────────────────┘

STORAGE TIER 1: PARQUET DATA (Persistent)
┌──────────────────────────────┐
│  /mnt/blob/datasets/c4/      │
│  ├─ file_0000.parquet        │
│  ├─ file_0001.parquet        │  Distributed across workers
│  ├─ file_0002.parquet        │  Each worker picks ONE random file
│  └─ file_XXXX.parquet        │
└──────────────────────────────┘
                │
                │ Random selection
                │
                ▼
        ┌───────────────┐
        │ Worker Picks  │
        │ One File      │
        └───────────────┘
                │
                │ pd.read_parquet()
                │
                ▼
STORAGE TIER 2: IN-MEMORY DATA (Worker Process)
┌──────────────────────────────┐
│ DataFrame (rows × columns)   │
│ ├─ 1000s of rows per file    │ Pick ONE random row
│ ├─ 'text' column             │
│ └─ Metadata columns          │ Text content
└──────────────────────────────┘
                │
                │ df.iloc[row_idx]["text"]
                │
                ▼
        ┌───────────────────┐
        │ Selected Text Row │
        │ (~1000 chars)     │
        └───────────────────┘


STORAGE TIER 3: MODEL CACHE (Shared, Read-Only)
┌──────────────────────────────┐
│  /mnt/blob/checkpoints/      │
│  base_models/gpt2-large/     │
│  ├─ config.json              │
│  ├─ pytorch_model.bin (2GB)  │ Cached by head node
│  ├─ tokenizer.json           │
│  └─ special_tokens_map.json  │
└──────────────────────────────┘
                │
                │ Load once per worker
                │
                ▼
    ┌─────────────────────┐
    │ Model in Memory     │
    │ (GPT2-Large)        │
    │ ~124M parameters    │
    │ ~2GB float16        │
    └─────────────────────┘


STORAGE TIER 4: WORKER CHECKPOINTS (Write Output)
┌──────────────────────────────┐
│  /mnt/blob/checkpoints/      │
│  worker_checkpoints/gpt2-l/  │
│  ├─ worker_000_checkpoint.pt │
│  ├─ worker_001_checkpoint.pt │ One per worker
│  ├─ worker_002_checkpoint.pt │ Contains model state + metadata
│  └─ worker_NNN_checkpoint.pt │
└──────────────────────────────┘
                │
                │ Average all states
                │
                ▼
STORAGE TIER 5: FINAL MODEL (Consolidation Output)
┌──────────────────────────────┐
│  /mnt/blob/checkpoints/      │
│  model/gpt2-large/           │
│  ├─ config.json              │
│  ├─ pytorch_model.bin        │ Averaged weights
│  ├─ tokenizer.json           │
│  └─ final_model_state.pt     │
└──────────────────────────────┘
```

---

## Metrics Collection & Aggregation Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     METRICS PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────┘

WORKER-LEVEL METRICS (Computed on Each Worker)
┌────────────────────────────────────────────────┐
│ Worker 0 Metrics Dict:                         │
│ {                                              │
│   "loss": 2.5,                                 │
│   "rank": 0,                                   │
│   "world_size": 20,                            │
│   "parquet_rows": 2487,                        │
│   "parquet_load_time": 0.0234,                 │
│   "parquet_size_mb": 15.6,                     │
│   "model_load_time": 3.4521,                   │
│   "model_size_mb": 256.3,                      │
│   "tokenizer_load_time": 0.0156,               │
│   "training_simulation_time": 10.0012,         │
│   "checkpoint_save_time": 2.1234,              │
│   "checkpoint_size_mb": 512.8,                 │
│   "overall_time": 26.1456                      │
│ }                                              │
└────────────────────────────────────────────────┘
                      │
                      │ ray.train.report()
                      │
                      ▼
┌────────────────────────────────────────────────┐
│ Worker 1 Metrics Dict (Same Structure)         │
└────────────────────────────────────────────────┘
                      │
                      │ ...20 workers total
                      │
                      ▼
HEAD NODE AGGREGATION
┌─────────────────────────────────────────────────────────────────┐
│ Collect all worker metrics into lists:                          │
│                                                                 │
│ parquet_load_time[]      = [0.0234, 0.0198, ..., 0.0276]       │
│ model_load_time[]        = [3.4521, 3.4712, ..., 3.4456]       │
│ checkpoint_save_time[]   = [2.1234, 2.0987, ..., 2.1456]       │
│ overall_time[]           = [26.1456, 26.2341, ..., 26.0987]    │
│ parquet_size_mb[]        = [15.6, 16.2, ..., 15.8]             │
│ model_size_mb[]          = [256.3, 256.3, ..., 256.3]          │
│ checkpoint_size_mb[]     = [512.8, 513.1, ..., 512.5]          │
│ parquet_rows[]           = [2487, 2401, ..., 2563]             │
│ losses[]                 = [2.5, 2.6, ..., 2.8]                │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
STATISTICS COMPUTATION
┌─────────────────────────────────────────────────────────────────┐
│ For Each Metric:                                                │
│ ├─ Average = SUM / COUNT                                       │
│ ├─ Min = MINIMUM(values)                                       │
│ ├─ Max = MAXIMUM(values)                                       │
│ └─ Sum = SUM(values) [For row counts]                          │
│                                                                 │
│ Example for parquet_load_time:                                 │
│ ├─ Average: 0.0234 sec                                         │
│ ├─ Min: 0.0156 sec (fastest worker)                            │
│ └─ Max: 0.0312 sec (slowest worker)                            │
│                                                                 │
│ Example for row counts:                                        │
│ ├─ Total Rows: 2487 + 2401 + ... = 50,342 rows                │
│ ├─ Average: 50,342 / 20 = 2,517 rows per worker               │
│ ├─ Min: 2,387 rows                                             │
│ └─ Max: 2,631 rows                                             │
└─────────────────────────────────────────────────────────────────┘
                      │
                      ▼
HUMAN-READABLE OUTPUT
┌─────────────────────────────────────────────────────────────────┐
│ Worker Timing Statistics (Average):                             │
│ ├─ Parquet Load:      0.0234s (min: 0.0156s, max: 0.0312s)     │
│ ├─ Model Load:        3.4521s (min: 3.4456s, max: 3.4712s)     │
│ ├─ Checkpoint Save:   2.1234s (min: 2.0987s, max: 2.1456s)     │
│ └─ Overall Time:     26.1456s (min: 26.0987s, max: 26.2341s)   │
│                                                                 │
│ Worker Data Size Statistics (Average):                          │
│ ├─ Parquet File:     15.6 MB (min: 15.2 MB, max: 16.8 MB)      │
│ ├─ Model:           256.3 MB (min: 256.3 MB, max: 256.3 MB)    │
│ └─ Checkpoint:      512.8 MB (min: 512.5 MB, max: 513.1 MB)    │
│                                                                 │
│ Worker Row Count Statistics:                                    │
│ ├─ Total Rows:       50,342 rows                               │
│ ├─ Avg per Worker:    2,517 rows (min: 2,387, max: 2,631)     │
│ └─ Workers:           20 total                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Configuration & Environment Variables

```
┌─────────────────────────────────────────────────────────────────────────┐
│               CONFIGURATION PARAMETERS (Via Environment)                │
└─────────────────────────────────────────────────────────────────────────┘

DATA CONFIGURATION
├─ DATA_DIR
│  └─ Default: "/mnt/blob/datasets"
│     Purpose: Root directory for parquet files
│
├─ PARQUET_PATH (computed)
│  └─ Pattern: "{DATA_DIR}/c4/*.parquet"
│     Purpose: Glob pattern to find all parquet files
│
├─ CONTAINER_NAME
│  └─ Storage container name
│

MODEL CONFIGURATION
├─ MODEL_NAME
│  └─ Default: "openai-community/gpt2-large"
│     Options: Any HuggingFace model identifier
│     Purpose: Which model to download and fine-tune
│
├─ HF_TOKEN
│  └─ HuggingFace API token for authentication
│     Purpose: Download private/gated models
│
├─ MAX_SEQ_LENGTH
│  └─ Default: 512
│     Purpose: Maximum token sequence length
│

TRAINING CONFIGURATION
├─ NUM_WORKERS
│  └─ Default: 2 (locally), Actual: 20 (cluster)
│     Purpose: Number of distributed workers
│
├─ LEARNING_RATE
│  └─ Default: 2e-5
│     Purpose: Optimizer learning rate
│
├─ CHECKPOINT_DIR
│  └─ Default: "/mnt/blob/checkpoints"
│     Purpose: Where to save worker checkpoints & final model
│

STORAGE PATHS (Computed)
├─ MODEL_CACHE_DIR
│  └─ {CHECKPOINT_DIR}/base_models/{MODEL_NAME}
│     Purpose: Cache location for pre-downloaded model
│
├─ WORKER_CHECKPOINT_DIR
│  └─ {CHECKPOINT_DIR}/worker_checkpoints/{MODEL_NAME}
│     Purpose: Where individual worker checkpoints saved
│
└─ TRAINED_MODEL_DIR
   └─ {CHECKPOINT_DIR}/model/{MODEL_NAME}
      Purpose: Final consolidated model location
```

---

## Key Functions & Their Roles

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      FUNCTION RESPONSIBILITIES                          │
└─────────────────────────────────────────────────────────────────────────┘

1. cache_model_locally(model_name, cache_dir, hf_token)
   ├─ Execution: HEAD NODE ONLY (before training)
   ├─ Purpose: Download model from HuggingFace & cache locally
   ├─ Steps:
   │  ├─ Check if model already cached (config.json exists)
   │  ├─ If not: Download model using HuggingFace API
   │  ├─ Download tokenizer
   │  └─ Save both to cache_dir
   ├─ Output: Cached model ready for workers to load
   └─ Why: Model is large (2GB), better to cache once than download per worker

2. train_loop_per_worker(config)
   ├─ Execution: EACH WORKER (in parallel)
   ├─ Purpose: Execute complete training simulation on single worker
   ├─ Steps:
   │  ├─ Get rank (worker ID) and world_size (total workers)
   │  ├─ Randomly select ONE parquet file from config["parquet_files"]
   │  ├─ Load parquet with pd.read_parquet()
   │  ├─ Randomly select ONE row
   │  ├─ Load model from cache
   │  ├─ Load tokenizer from cache
   │  ├─ Simulate training with time.sleep(10)
   │  ├─ Synchronize all workers at barrier
   │  ├─ Save checkpoint with model state
   │  ├─ Compile metrics dict with all timings/sizes
   │  └─ Report metrics via ray.train.report()
   ├─ Output: Checkpoint file + reported metrics
   └─ Why: Simulates real distributed training flow

3. consolidate_checkpoints(checkpoint_dir, model_name)
   ├─ Execution: HEAD NODE ONLY (after all workers complete)
   ├─ Purpose: Average all worker model states into final model
   ├─ Steps:
   │  ├─ Find all worker_*.pt checkpoint files
   │  ├─ Load model state dict from each checkpoint
   │  ├─ Average all state dicts: mean(stack(states))
   │  ├─ Load base model and apply averaged state
   │  ├─ Save final model to TRAINED_MODEL_DIR
   │  └─ Save metadata file with losses & averaging info
   ├─ Output: Final consolidated model
   └─ Why: Simulate federated averaging (averaging all worker improvements)

4. main block (if __name__ == "__main__")
   ├─ Execution: HEAD NODE ONLY
   ├─ Orchestration:
   │  ├─ Initialize Ray cluster
   │  ├─ Create checkpoint directory structure
   │  ├─ Cache model locally
   │  ├─ Discover all parquet files
   │  ├─ Configure TorchTrainer with NUM_WORKERS
   │  ├─ Launch distributed training (trainer.fit())
   │  ├─ Collect & aggregate worker metrics
   │  ├─ Print summary statistics
   │  ├─ Consolidate worker checkpoints
   │  ├─ Shutdown Ray
   │  └─ Save final model
   ├─ Output: Trained model + checkpoint files + metrics
   └─ Why: Main orchestration point for entire pipeline
```

---

## Performance Characteristics & Bottlenecks

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   PERFORMANCE ANALYSIS                                  │
└─────────────────────────────────────────────────────────────────────────┘

TIMING BREAKDOWN (Per Worker)

Head Node Timeline:
├─ Model Caching:              ~30-60 sec (one-time, parallel)
├─ Checkpoint Dir Setup:       ~0.1 sec
├─ Parquet Discovery:          ~1-5 sec (glob)
├─ Training Launch:            ~2-5 sec (Ray overhead)
├─ Wait for Workers:           ~26 sec (parallel execution)
├─ Metrics Collection:         ~0.1 sec
├─ Checkpoint Consolidation:   ~5-10 sec (average states)
└─ Total Head Node Time:       ~50-90 sec

Worker Timeline (Each in Parallel):
├─ Random File Selection:      ~0.001 sec
├─ Parquet Load:               ~0.02-0.05 sec
├─ Random Row Selection:       ~0.001 sec
├─ Model Load:                 ~3.4-3.5 sec ◄── Dominant
├─ Tokenizer Load:             ~0.02 sec
├─ Training Simulation:        ~10.0 sec ◄── Dominant
├─ Worker Sync (barrier):      ~0.01 sec (all workers wait)
├─ Checkpoint Save:            ~2.0-2.2 sec
├─ Metrics Report:             ~0.01 sec
└─ Total Per-Worker Time:      ~25.5 sec

CRITICAL OBSERVATIONS:
1. Model Load (3.4s) + Training Sim (10s) = 13.4s / 25.5s = 52% of time
2. All workers run in parallel → Total Training Time ≈ 26 sec (not 26×20=520s)
3. Head node model caching is SERIALIZED but shared (done once)
4. Barrier sync ensures all workers finish before checkpoint save
5. Worker metrics collection is asynchronous (Ray handles)


RESOURCE UTILIZATION:

Head Node:
├─ CPU: Low (orchestration only)
├─ Memory: ~2GB (one model instance)
└─ I/O: Moderate (checkpoint read/write)

Each Worker:
├─ CPU: Low (simulated training)
├─ Memory: ~2.5-3GB (model + data)
│  ├─ Model: ~2GB
│  ├─ Parquet data: ~16MB
│  └─ Checkpoints: ~512MB (temporary, saved)
└─ I/O: Low (minimal read/write outside init)

Cluster Total (20 workers):
├─ CPU Cores: ~20 cores actively used (one per worker)
├─ Memory: ~50-60GB (20 workers × 2.5-3GB)
└─ Storage I/O: Bounded by checkpoint save (network bottleneck)
```

---

## Optimization Opportunities

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   CURRENT OPTIMIZATIONS IN PLACE                       │
└─────────────────────────────────────────────────────────────────────────┘

✅ IMPLEMENTED OPTIMIZATIONS:

1. Worker-Computed Statistics
   ├─ What: Each worker computes its own metrics (rows, sizes, timing)
   ├─ Why: Eliminates expensive head node I/O
   ├─ Impact: Training starts instantly (no pre-computation delay)
   └─ Code: len(df), os.path.getsize(), time.time()

2. Lazy Model Caching
   ├─ What: Model cached once, reused by all workers
   ├─ Why: Avoid 20× redundant downloads
   ├─ Impact: ~30-60 sec saved vs. per-worker caching
   └─ Code: check config.json, only download if missing

3. Random File Selection
   ├─ What: Each worker picks ONE random parquet file
   ├─ Why: Distribute data uniformly across workers
   ├─ Impact: Better load balancing
   └─ Code: random.choice(parquet_files)

4. Worker Synchronization (Barrier)
   ├─ What: All workers wait at barrier before checkpoint save
   ├─ Why: Ensure all training complete before consolidation
   ├─ Impact: Data consistency, avoid race conditions
   └─ Code: dist.barrier()

5. Local Checkpoint Aggregation
   ├─ What: Metrics collected in lists, not individual I/O
   ├─ Why: Batch processing faster than per-worker I/O
   ├─ Impact: Sub-second aggregation time
   └─ Code: timings[], sizes[], row_counts[]


⚠️ POTENTIAL FUTURE OPTIMIZATIONS:

1. Async Checkpoint Saving
   ├─ Why: Save checkpoints in background, don't block workers
   └─ Trade-off: Complexity increase

2. Gradient Compression
   ├─ Why: Reduce checkpoint file sizes (currently 512MB)
   └─ Trade-off: Adds compression/decompression overhead

3. Model Quantization
   ├─ Why: Reduce model size from 2GB to ~500MB (int8)
   └─ Trade-off: Accuracy loss, might break simulation

4. Distributed Checkpointing
   ├─ Why: Save checkpoints to worker local storage, not shared
   └─ Trade-off: Adds consolidation complexity

5. Batch Processing
   ├─ Why: Process multiple rows per worker instead of one
   └─ Trade-off: Changes simulation semantics
```

---

## Summary Table

| Component | Type | Location | Role |
|-----------|------|----------|------|
| **Head Node** | Process | Cluster Master | Orchestration, model caching, metrics collection |
| **Worker (×20)** | Process | Ray Worker Pod | Training simulation, checkpoint save, metrics reporting |
| **Model Cache** | File | `/mnt/blob/checkpoints/base_models/` | Shared model weights (2GB, read-only) |
| **Parquet Data** | File | `/mnt/blob/datasets/c4/` | Input training data (random sample per worker) |
| **Worker Checkpoints** | File | `/mnt/blob/checkpoints/worker_checkpoints/` | Per-worker model states (512MB × 20) |
| **Final Model** | File | `/mnt/blob/checkpoints/model/` | Consolidated/averaged model output |
| **Metrics** | Dict | Ray Train | Per-worker timing, size, and row count statistics |

---

## How It Differs From Real Training

```
THIS SIMULATION              │  REAL TRAINING
────────────────────────────┼──────────────────────────
Random row per worker       │  Fixed batch per worker
10-second sleep             │  Actual forward/backward pass
Simulated loss (2.5+rank×0.1)│ Computed from model output
No gradient computation     │  Backprop through model
No actual loss.backward()   │  Real gradients computed
Average all states naively  │  Sophisticated federated averaging
────────────────────────────┼──────────────────────────
PURPOSE: Test infrastructure│  PURPOSE: Actually train models
SCOPE: Distributed pipeline │  SCOPE: ML optimization
```

---

## Key Takeaways

**What train_minimal.py Does:**
1. **Simulates** distributed training with real Ray infrastructure
2. **Tests** checkpoint save/load and metrics collection
3. **Measures** performance across workers (timing, data sizes, row counts)
4. **Demonstrates** federated learning pattern (average all worker states)
5. **Validates** cluster setup before running real training

**Why It's Useful:**
- Fast feedback loop (26 sec total vs. hours of real training)
- Tests networking, storage, and checkpoint I/O
- Collects baseline performance metrics
- Validates configuration before scaling to real workload
- Allows tuning of NUM_WORKERS, timing, checkpoint locations

**Critical Path Items:**
1. Model loading: 3.4s per worker (serialized)
2. Training simulation: 10s per worker (parallel)
3. Checkpoint save: 2.1s per worker (parallel)
4. **Total: ~26 sec with 20 workers (no head node blocking)**
