# Event extractor contract

The first implementation is deterministic. A future local model must return only
the structured fields represented by `ExtractedEvent` and must never receive an
excluded-app event or an unbounded capture payload.

