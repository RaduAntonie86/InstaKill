# Week 7 feed evaluation protocol

**Question.** On the seeded database, how well do learned (three signals), for_you (six-signal Week 5 neural model), popularity, and chronological ranking place posts users like later? We report NDCG@10 for each eligible user and compare for_you with the strongest simple baseline.

**Cut-off and snapshot.** Unless `--cutoff` is supplied, the cut-off is the timestamp at the 80th percentile of likes, using the higher observed value. The evaluator backs up SQLite into a temporary database, then removes likes, follows, and posts created at or after the cut-off. Historical MovieLens follows are not synthesized from likes by this repository's seeder; any stored follow is retained only when its own timestamp precedes the cut-off. The live database is read-only throughout.

**Users.** A user is included when at least three of their likes occurred before the cut-off and they have at least one later like on a post present in the snapshot. Activity groups use their count of pre-cut-off likes: new (<10), casual (10–49), regular (50–199), and power (200+).

**Candidates.** Each strategy receives the same posts that existed before the cut-off, the same pre-cut-off like history, and `now` fixed to the cut-off. The evaluator invokes `app.application.feed.rank_feed`, the same service used by `GET /feed`. Learned scores the first three Week 5 signals with the Week 4 weights; for_you runs the committed `like_model.npz` over all six signals. Popularity orders by the product's last-365-day popularity signal.

**Relevance.** Relevant items for a user are the snapshot posts that user liked at or after the cut-off. These later interactions are used only as labels and never enter the snapshot ranking history.

**Metric.** NDCG@10 uses binary relevance. DCG sums `1/log2(rank+1)` for relevant results at zero-based positions 0–9, then divides by the ideal DCG with all available relevant posts first. Users with no eligible later relevant posts are not included.

**Intervals.** We compute 95% percentile bootstrap intervals over users with replacement, using the requested fixed seed (default 42) and 5,000 resamples. The for_you comparison is paired: each resample draws users once and subtracts the selected best baseline's score from for_you for those same users. The baselines are learned, popularity, and chronological. A difference is a win if the interval is above zero, a loss if below zero, and unclear otherwise.

**Slices.** Results include the four required pre-cut-off activity groups. The overall and per-group best baseline is selected by its mean in that same population.

**Known limitations.** The for_you model was trained on the Week 5 MovieLens data and may have seen events in this evaluation period. `like_model.npz` contains no training cutoff, so the precise overlap cannot be recovered; the report counts eligible future likes as potential overlap (46 in the checked-in run), and for_you results should be read as potentially optimistic. The database population is MovieLens ratings summarized as old movie posts, rather than a social feed of fresh posts; conclusions for a feed of fresh posts should therefore be checked on fresh-post data rather than generalized from this replay.

These numbers are not comparable to the lab's moment-level scores: the lab hides one liked movie among 99 random candidates at each moment, while this replay ranks the full available catalogue and counts every later like as relevant.
