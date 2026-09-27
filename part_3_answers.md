# Part 3 — Problem-Solving Answers

### 1. Your classifier turns out to be wrong about 30% of the time. What do you do and how would you even decide whether it's "good enough" to be useful at all?

First, I'd want to know *how* it's wrong, not just how often. A flat 30% accuracy drop is just a high-level metric, but a confusion matrix tells the actual story. If the model is wrong 30% of the time because it consistently confuses "Highway" with "Residential" but nails everything else, it is still a highly useful tool—we just need to tune features or gather more data for those two classes. If the errors are completely random across all land-use types, the model is genuinely weak.

Whether it is "good enough" isn't a single number; it depends entirely on what the output feeds into. Since this system is designed for an analyst to query results entirely offline, a 30% error rate is perfectly fine *if* confidence correlates with correctness. If the model successfully flags its uncertain predictions for human review (which is how I designed the pipeline in Part 1), the system works as intended. The low-confidence tiles are disproportionately the wrong ones, so the analyst only spends time on the hard cases. I would plot accuracy against confidence buckets to verify this. If the model is confidently incorrect, the triage system fails, and no amount of downstream filtering can fix that.

### 2. This service runs offline, with no one watching it live. A month after deployment, how would you know it's still working correctly?

Since there's no live dashboard to glance at, the system needs baked-in, automated health checks that run completely offline.

First, I would embed a small "canary set" of satellite image tiles with known ground-truth labels directly on the hardware. A cron job would run this static set through the pipeline on a schedule—say, weekly—and compare the outputs against the expected labels. If accuracy on this set drifts over a month, we immediately know it's a real system issue (like model corruption, a silent preprocessing bug, or drive issues) rather than just a hunch.

Second, I'd track the shape of the output distribution over time. Even without ground truth, if the historical ratio of tiles landing in the `FLAGGED_REVIEW` queue versus the `COMPLETED` queue suddenly spikes, or if the model abruptly classifies everything as "Industrial," it's a strong sign something broke. Both the canary results and these distribution stats can be logged locally to a simple, lightweight text report. An operator can quickly glance at this log periodically to verify system health without needing any live monitoring infrastructure.

### 3. Tiles are coming in fine, but the stored results look wrong. Walk us through how you'd find the cause—your actual steps, in order.

I troubleshoot starting with the cheapest, most mechanical checks before assuming the model is at fault.

1. **Data Integrity:** I pull a specific bad result, get its `checksum_sha256` from the metadata row, and re-hash the actual file on disk. If they don't match, it's not a model problem at all—it's disk corruption or a race condition during the write process. I'd stop looking at the machine learning side and go check the artifact storage layer instead.
2. **Preprocessing Validation:** If the checksum is intact, I'd manually run that exact stored image through the preprocessing steps by hand (outside the API, in a plain script). I'd inspect the output tensor's shape, value ranges, and normalization. A silent preprocessing bug (like a swapped band order, wrong resize interpolation, or outdated normalization constants) is the single most common way results look "wrong" even though the model itself is fine.
3. **Pipeline Handoff:** If preprocessing looks correct, I feed that exact tensor directly into the model, bypassing the API and database entirely, and compare the raw output against what got stored. If they differ, something is wrong in how the result gets written back—a serialization bug, a class-index mapping that's off by one, or writing to the wrong column in the database update.
4. **Model Reality Check:** Only if all of that lines up and the model output itself is just bad do I conclude that it's a genuine model-quality problem. Assuming "the model is just wrong" first is a trap that is both the least actionable path and the easiest way to miss standard software bugs.

### 4. What's the weakest part of your design, and what would break it first?

The confidence threshold itself. In my design, handling uncertainty relies heavily on a tuned numerical threshold to flag low-confidence predictions for human review. The entire correctness-under-uncertainty story in Part 1—the whole reason a 30%-wrong model is still usable—depends on that threshold actually correlating with whether the model is right.

If the model is poorly calibrated (confidently wrong as often as it's confidently right) on real-world, out-of-distribution data, the entire `FLAGGED_REVIEW` queue stops doing its job. Either everything gets flagged (defeating the purpose of automation and causing analysts to just rubber-stamp things) or nothing does (dangerous, incorrect predictions sail through to the `COMPLETED` queue looking just as trustworthy as accurate ones). That is the first thing that would break, and it would break quietly—nothing crashes, no error gets logged, the data just quietly loses its operational value. I'd want the canary-set idea from question 2 running from day one specifically to catch this.