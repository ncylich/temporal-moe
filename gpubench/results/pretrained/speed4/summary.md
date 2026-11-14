| arm | setting | texts | tok/s | range over texts | ms per token | loads per decoded token-layer | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| base | vram | 5 | 655.7 | 654.7 to 656.1 | 1.525 | -- | 1438 |
| ours_stream | k2-x0 | 5 | 585.4 | 584.1 to 587.5 | 1.708 | 1.000 | 918 |
| ours_stream | k2-x1 | 5 | 584.1 | 579.4 to 586.5 | 1.712 | 1.000 | 918 |
| ours_stream | s0x0 | 5 | 602.8 | 599.6 to 603.9 | 1.659 | 0.999 | 918 |
| ours_stream | s0x1 | 5 | 601.4 | 595.1 to 604.5 | 1.663 | 0.999 | 918 |
| ours_stream | s1x0 | 5 | 615.7 | 608.6 to 619.1 | 1.624 | 0.999 | 918 |
| ours_stream | s1x1 | 5 | 614.9 | 611.0 to 617.5 | 1.626 | 0.999 | 918 |
| ours_stream | s2x0 | 5 | 616.2 | 609.0 to 618.7 | 1.623 | 0.999 | 918 |
| ours_stream | s2x1 | 5 | 616.3 | 615.3 to 616.8 | 1.622 | 0.999 | 918 |
| ours_vram | k2 | 5 | 672.3 | 671.9 to 672.6 | 1.487 | 1.000 | 1438 |
| ours_vram | vram | 5 | 699.7 | 698.3 to 700.4 | 1.429 | 0.999 | 1438 |
| skliar_stream | fullsel-x0 | 5 | 549.6 | 544.9 to 553.8 | 1.819 | 1.055 | 918 |
| skliar_stream | fullsel-x1 | 5 | 550.2 | 543.9 to 556.2 | 1.818 | 1.055 | 918 |
| skliar_stream | s0x0 | 5 | 581.4 | 573.7 to 587.5 | 1.720 | 1.059 | 918 |
| skliar_stream | s0x1 | 5 | 580.7 | 568.8 to 587.7 | 1.722 | 1.059 | 918 |
| skliar_stream | s1x0 | 5 | 591.2 | 583.8 to 600.0 | 1.691 | 1.059 | 918 |
| skliar_stream | s1x1 | 5 | 592.8 | 589.0 to 597.7 | 1.687 | 1.059 | 918 |
| skliar_stream | s2x0 | 5 | 593.6 | 585.7 to 596.8 | 1.685 | 1.059 | 918 |
| skliar_stream | s2x1 | 5 | 591.2 | 583.7 to 598.8 | 1.692 | 1.059 | 918 |
| skliar_stream | sk-k2-s1x1 | 5 | 581.2 | 575.6 to 586.2 | 1.721 | 1.059 | 918 |
| skliar_vram | fullsel | 5 | 644.8 | 643.8 to 646.5 | 1.551 | 1.057 | 1438 |
| skliar_vram | sk-k2 | 5 | 673.4 | 671.2 to 675.0 | 1.485 | 1.054 | 1438 |
| skliar_vram | vram | 5 | 684.9 | 683.9 to 685.9 | 1.460 | 1.054 | 1438 |
| trained_stream | s0x0 | 5 | 585.4 | 584.7 to 585.8 | 1.708 | 0.999 | 918 |
| trained_stream | s0x1 | 5 | 583.9 | 578.7 to 587.4 | 1.713 | 0.999 | 918 |
| trained_stream | s1x0 | 5 | 598.2 | 597.5 to 599.2 | 1.672 | 0.999 | 918 |
| trained_stream | s1x1 | 5 | 598.5 | 597.1 to 599.7 | 1.671 | 0.999 | 918 |
| trained_stream | s2x0 | 5 | 598.4 | 597.7 to 600.0 | 1.671 | 0.999 | 918 |
| trained_stream | s2x1 | 5 | 598.2 | 596.9 to 599.4 | 1.672 | 0.999 | 918 |
| trained_vram | vram | 5 | 677.2 | 676.0 to 678.1 | 1.477 | 0.999 | 1438 |

Each streamed method at its fastest setting:

| arm | fastest setting | tok/s | ms per token |
|---|---|---|---|
| ours_stream | s2x1 | 616.3 | 1.622 |
| skliar_stream | s2x0 | 593.6 | 1.685 |
| trained_stream | s1x1 | 598.5 | 1.671 |
