# Fine-tuning Reinforcement Learning Models is Secretly a Forgetting Mitigation Problem

> **Placeholder.** Drop in the official PaperBench-converted Markdown of
> Wolczyk et al. (ICML 2024 Spotlight). See `third_party/paperbench/README.md`
> for the swap-in process.

## Abstract (paraphrased synopsis, not quoted)

The paper studies fine-tuning of reinforcement-learning policies under
distribution shift and shows that what looks like a fine-tuning problem is
better framed as catastrophic-forgetting mitigation, with concrete remedies
drawn from the continual-learning literature.

## Why this paper for our head-to-head

- ICML 2024 Spotlight in Deep RL.
- 178 rubric leaves (smallest viable rubric across non-GPU PaperBench papers).
- Published BasicAgent baselines: Claude 3.5 Sonnet 9.3% ± 1.0, o1 1.7% ± 0.8.
- Brax / JAX based; runs on CPU.

## Where to find the real text

OpenReview: https://openreview.net/forum?id=53iSXb1m8w
