# CLUSTER.md — training hardware

Fine-tuning runs on the NUS SoC Compute Cluster. This records the durable facts
and the decisions taken; live node availability changes hourly and is
deliberately not captured here.

## Access

- Login node: `xlogin.comp.nus.edu.sg`
- Jobs are submitted through **Slurm**, not run interactively.

## Partitions

| partition | max job time |
|---|---|
| `gpu` | 3 hours |
| `gpu-long` | 3 days |

## GPUs available

| type | VRAM | billing weight |
|---|---:|---:|
| A100-40 | 40 GB | 9 |
| A100-80 | 80 GB | 12 |
| H100-47 | 47 GB | 12 |
| H100-96 | 96 GB | 24 |
| H200-141 | 141 GB | 48 |
| `nv` (unspecified) | ? | — |

`nv` nodes report only `gpu:nv` and are not identified by model; do not assume
they are A100-class.

Check live availability before submitting:

```bash
sinfo -p gpu-long -o "%N %G %T"
```

Do not hard-code node names. Let Slurm schedule.

## Decisions

### A100-40, on the `gpu` partition

Both choices optimise for **queue time rather than speed**, because a run takes
roughly 25 minutes and waiting for a slot will dominate it.

- **A100-40** is the smallest GPU that fits the job (see the memory budget
  below). Billing weight is the reason to care: taking an H200 at weight 48 for
  a job needing 20 GB burns five times the fair-share allocation for headroom
  that goes unused, degrading queue priority on every later job. With a dozen
  sweep runs planned, that compounds.
- **`gpu`, not `gpu-long`.** A 25-minute run fits comfortably inside the
  three-hour cap, and short-job partitions drain faster. `gpu-long` is for a
  full sweep submitted as one job, or for 32B experiments.

H100 is roughly 2-3x faster, which would turn a 25-minute run into 10 minutes.
Not worth a higher billing weight and a longer queue at this scale.

### Plain LoRA, not QLoRA

Memory budget for an 8B model in bf16:

| | |
|---|---:|
| model weights (8B x 2 bytes) | 16 GB |
| LoRA adapters, gradients, optimiser states | ~0.5 GB |
| activations (gradient checkpointing, seq 512) | ~4 GB |
| **total** | **~20 GB** |

That fits in 40 GB with room to spare, so there is no reason to accept QLoRA's
costs: 4-bit quantisation loses precision, and unpacking weights on every
forward pass runs roughly 20-40% slower.

QLoRA becomes the right choice at 14B+ on this card, or for 32B anywhere. The
GPU ladder then maps onto the model ladder: A100-40 for 8B, A100-80 or H100-96
for larger.

## Before the first submission

**Check whether compute nodes have outbound internet.** They frequently do not,
in which case `from_pretrained` cannot reach HuggingFace and the job fails at
startup. If so, download weights on the login node into shared storage and point
`HF_HOME` at that path from the sbatch script.

Also confirm the expected environment setup (`module avail`), which is
site-specific.

## Still open

- base model and size
- training framework
- hyperparameters

See [PROJECT.md](PROJECT.md).
