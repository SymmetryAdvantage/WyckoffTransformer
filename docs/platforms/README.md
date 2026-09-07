# Platform-specific setup

WyFormer is run on a range of machines, and their environments differ in ways that
do not generalise: GPU driver versions, available container runtimes, filesystem
layout, schedulers, and local package indices.

Matching scripts live in `scripts/platforms/<platform>/`.

If you are setting up a host that already has a directory here, follow it rather
than improvising.
