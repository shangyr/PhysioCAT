# Submitted figure sources

Figures 1--2 use editable diagram files in `diagrams/`. Figure 4 uses a publication-facing vector base for the schematic A/F panels; its fixed-window B/C matrices are released in `data/`, Panel C is materialized from the released no-delay matrix, and data-driven D/E are refreshed by `render_figure_4.py`. Figures 3, 5, and Supplementary Figures S1--S2 (stored internally as Figure_6--7) are regenerated from released predictions, metrics, panel manifests, and display sources. Scientific values, extracted text, page geometry, and fixed-raster content must agree; byte identity is also reported when the renderer environment matches.

The Matplotlib-rendered figures use the bundled DejaVu Sans family so text metrics and panel geometry remain reproducible across Windows and Linux without proprietary system fonts.
