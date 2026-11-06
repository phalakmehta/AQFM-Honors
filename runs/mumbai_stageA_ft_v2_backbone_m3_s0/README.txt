Stage A run (2026-11-06 16:09-16:13): v2 + freeze backbone, 3 months, seed 0, lr 1e-4 (finetune default), 3000 steps,
warmup 300, validation every 250, patience 4; command in runs/mumbai_transfer_A.log. It was written to
runs/mumbai_ft_v2_backbone_m3_s0, which stage B then reused: stage B appended to its log.jsonl and replaced its
config.json, best.pt and console log. log.jsonl here = the first 31 lines of that file (this run, steps 1-1250,
5 validations, best 1.9213 as printed in runs/mumbai_transfer_A.log); the rest stayed with the stage-B run.
