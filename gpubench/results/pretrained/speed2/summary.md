| arm | setting | texts | tok/s | range over texts | ms per token | loads per decoded token-layer | peak VRAM MiB |
|---|---|---|---|---|---|---|---|
| base | vram | 5 | 654.9 | 653.6 to 656.7 | 1.527 | -- | 1438 |
| ours_stream | k2-x0 | 5 | 584.7 | 578.7 to 587.0 | 1.710 | 1.000 | 918 |
| ours_stream | k2-x1 | 5 | 584.7 | 580.4 to 587.0 | 1.710 | 1.000 | 918 |
| ours_stream | s0x0 | 5 | 600.9 | 595.6 to 605.2 | 1.664 | 0.999 | 918 |
| ours_stream | s0x1 | 5 | 599.1 | 594.7 to 603.5 | 1.669 | 0.999 | 918 |
| ours_stream | s1x0 | 5 | 612.0 | 609.3 to 617.7 | 1.634 | 0.999 | 918 |
| ours_stream | s1x1 | 5 | 610.5 | 609.8 to 611.8 | 1.638 | 0.999 | 918 |
| ours_stream | s2x0 | 5 | 615.2 | 607.6 to 618.8 | 1.625 | 0.999 | 918 |
| ours_stream | s2x1 | 5 | 612.7 | 608.8 to 617.7 | 1.632 | 0.999 | 918 |
| ours_vram | k2 | 5 | 672.9 | 671.0 to 673.7 | 1.486 | 1.000 | 1438 |
| ours_vram | vram | 5 | 700.0 | 699.2 to 701.2 | 1.429 | 0.999 | 1438 |
| skliar_stream | fullsel-x0 | 5 | 550.7 | 546.0 to 556.6 | 1.816 | 1.055 | 918 |
| skliar_stream | fullsel-x1 | 5 | 550.9 | 545.1 to 557.1 | 1.815 | 1.055 | 918 |
| skliar_stream | s0x0 | 5 | 569.5 | 563.5 to 578.1 | 1.756 | 1.059 | 918 |
| skliar_stream | s0x1 | 5 | 569.7 | 562.9 to 576.7 | 1.755 | 1.059 | 918 |
| skliar_stream | s1x0 | 5 | 578.4 | 572.2 to 583.9 | 1.729 | 1.059 | 918 |
| skliar_stream | s1x1 | 5 | 580.6 | 573.6 to 585.5 | 1.722 | 1.059 | 918 |
| skliar_stream | s2x0 | 5 | 577.7 | 572.2 to 581.5 | 1.731 | 1.059 | 918 |
| skliar_stream | s2x1 | 5 | 580.2 | 571.9 to 584.4 | 1.724 | 1.059 | 918 |
| skliar_vram | fullsel | 5 | 646.8 | 646.0 to 647.6 | 1.546 | 1.057 | 1438 |
| skliar_vram | vram | 5 | 671.9 | 671.2 to 672.2 | 1.488 | 1.054 | 1438 |
| trained_stream | s0x0 | 5 | 585.5 | 584.9 to 586.6 | 1.708 | 0.999 | 918 |
| trained_stream | s0x1 | 5 | 585.9 | 584.6 to 586.6 | 1.707 | 0.999 | 918 |
| trained_stream | s1x0 | 5 | 597.6 | 595.4 to 599.8 | 1.673 | 0.999 | 918 |
| trained_stream | s1x1 | 5 | 598.7 | 597.0 to 599.7 | 1.670 | 0.999 | 918 |
| trained_stream | s2x0 | 5 | 596.1 | 591.9 to 598.1 | 1.677 | 0.999 | 918 |
| trained_stream | s2x1 | 5 | 598.3 | 596.9 to 599.7 | 1.671 | 0.999 | 918 |
| trained_vram | vram | 5 | 678.1 | 676.4 to 679.9 | 1.475 | 0.999 | 1438 |

Each streamed method at its fastest setting:

| arm | fastest setting | tok/s | ms per token |
|---|---|---|---|
| ours_stream | s2x0 | 615.2 | 1.625 |
| skliar_stream | s1x1 | 580.6 | 1.722 |
| trained_stream | s1x1 | 598.7 | 1.670 |
