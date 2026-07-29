# Versioning and archival policy

[`main`](https://github.com/shangyr/PhysioCAT) is the maintained branch. The
state accompanying the initial manuscript submission is fixed by the annotated
tag [`bspc-submission-v1`](https://github.com/shangyr/PhysioCAT/tree/bspc-submission-v1)
and its corresponding [versioned GitHub Release](https://github.com/shangyr/PhysioCAT/releases/tag/bspc-submission-v1).
The Release metadata records the target commit SHA outside the tracked source
tree, while the manuscript cites the stable version, tag, and Release URL
without creating a self-referential commit identifier.

If a correction is required during peer review, it is committed normally,
validated by the complete reproduction workflow, and assigned a new immutable
tag such as `bspc-revision-v1`. Earlier reviewed tags are not moved, replaced,
or deleted. After acceptance, the accepted tag is archived in a DOI-issuing
repository such as Zenodo.
