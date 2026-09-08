# Design notes

This section preserves the long-form reasoning behind Tiga: architecture
decisions, IR sketches, scheduling research, and related systems.

!!! warning "Normative versus exploratory"

    These pages preserve design context and may discuss future work. The
    [Python API](api.md), [current status](roadmap.md), and the completion ledger
    in `PROJECT.md §15.3` define what is executable today. If a research note
    conflicts with those sources, it is not a supported feature.

## Reading order

Start with the [architecture RFC](rfcs/0001-architecture.md), then use the
[IR notes](IR_DESIGN.md) as a detailed reference. Read the
[scheduling](SCHEDULING_ABSTRACTIONS.md) and
[GPU optimization](GPU_GRAPH_OPTIMIZATION.md) notes when working on passes,
and [related work](RELATED_WORK.md) for research context.
