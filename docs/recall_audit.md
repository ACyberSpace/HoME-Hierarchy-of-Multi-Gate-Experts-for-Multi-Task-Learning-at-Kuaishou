# Recall Implementation Audit

## Shared Protocol

- Train history: days before the final test day.
- Test target: the final day positive union of `is_click`, `long_view`, `is_like`, `is_comment`, `is_forward`, and `is_follow`, deduplicated by `user_id, video_id`.
- Candidate output: every channel returns ranked `user_id -> [video_id]`.
- Evaluation: report per-channel and fused `MicroRecall@K`, `UserAvgRecall@K`, item coverage, overlap, and marginal contribution.

## Swing

Expected boundary:

- Input is user behavior sequences, usually click or positive feedback sequences.
- Build item similarity from user co-occurrence with user-pair penalties.
- Production systems cap long user histories, hot-item user lists, and retained top item neighbors to control memory.

Current implementation:

- Uses preprocessed `user_sequences["full_sequences"]` as the primary sequence source.
- Computes sparse item-pair common-user lists instead of a dense item-item matrix.
- Scores item pairs with capped common users and user-pair penalties:
  `weight(u) * weight(v) / (|I_u intersect I_v| + alpha2)`.
- Keeps only top similar items per source item.

Known approximation:

- This is an in-memory sparse approximation of the standard Swing idea, not a distributed PAI/MaxCompute implementation.
- Caps such as `max_user_items`, `max_user_per_item`, `max_pair_users`, and `max_sim_items` affect recall and memory.

## Item2Vec

Expected boundary:

- Train Word2Vec on user item behavior sequences.
- Retrieve nearest items from the embedding space, not by Python user-item double loops.

Current implementation:

- Uses preprocessed user behavior sequences.
- Uses gensim `KeyedVectors.most_similar` for vectorized top-N candidate retrieval.
- Filters already-seen history items from returned candidates.

Known approximation:

- User representation is gensim's averaged positive context vector over recent history items.
- This is exact for the local gensim implementation path, but not an ANN service.

## DSSM

Expected boundary:

- Two-tower retrieval model with user and item towers in a shared embedding space.
- Train with positive and sampled negative items.
- Precompute item tower embeddings and rank by dot product or cosine-like similarity.

Current implementation:

- Uses user and item sparse features.
- Negative sampling excludes user positive items where possible.
- Candidate generation uses full item feature tensors and precomputed item representations.

Known limitation:

- Loss is pointwise BCE over sampled positives/negatives, not sampled-softmax or in-batch softmax.

## MIND

Expected boundary:

- Multi-interest user representation, usually via dynamic routing/capsule-style interest extraction.
- Candidate retrieval takes the maximum score over multiple user interest vectors.

Current implementation:

- Produces multiple interest vectors from short behavior sequence embeddings.
- Scores items by maximum dot product over interests.
- Candidate generation uses full item feature tensors and item tower outputs.

Known limitation:

- Capsule module is a simplified projection-based approximation, not a full dynamic-routing MIND implementation.

## SDM

Expected boundary:

- Combines short-term sequential/session interest and long-term user behavior interest.
- Uses a gate to merge short and long interests for matching.

Current implementation:

- Short interest: BiLSTM + multi-head attention + projection.
- Long interest: multi-head attention over long sequence.
- User sparse feature tower is included in the gate input.
- Candidate generation uses full item feature tensors and item tower outputs.

Known limitation:

- This is a compact PyTorch approximation of SDM, not a full industrial SDM implementation with session-level auxiliary objectives.

## Freshness

Expected boundary:

- Non-personalized or lightly personalized recency recall.
- Acts as a complementary channel for new/fresh items.

Current implementation:

- Sorts `video_info` by `upload_timestamp` when available.
- Filters items already present in the user's history.

Known limitation:

- If raw upload time is not present in the selected video feature file, it falls back to encoded/video order and should be treated as a weak freshness proxy.
