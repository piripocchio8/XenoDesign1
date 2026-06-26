# XenoDesign1 README figure: de-novo non-alpha (cystine-knot / ICK-class) all-D binder
# Render: pymol -cq scripts/figures/render_non_alpha.pml
load XenoDesign1_local_ref/denovo_non_alpha/loop/iter_011/chai_out/pred.model_idx_0.cif, m

# --- shared XenoDesign1 figure style (do not change between figures) ---
bg_color white
set ray_opaque_background, 1
set ray_trace_mode, 1
set ray_trace_color, grey30
set antialias, 2
set ray_shadows, 0
set cartoon_fancy_helices, 1
set cartoon_highlight_color, grey60
set spec_reflect, 0.2
set ambient, 0.30
set direct, 0.55
set sphere_scale, 0.45
set dash_color, grey40
set dash_gap, 0.30
set dash_width, 2.5
set label_size, 18
set label_color, black
set label_outline_color, white
set float_labels, 1
set depth_cue, 0
# --- end shared style ---

hide everything
show cartoon

# faded two-chain target (context)
color grey85, (chain A or chain B)
set cartoon_transparency, 0.6, (chain A or chain B)

# designed all-D 30-mer binder
color salmon, chain C
select dres, (chain C and resn DAL+DAR+DSG+DAS+DCY+DGN+DGL+DHI+DIL+DLE+DLY+MED+DPN+DPR+DSN+DTH+DTR+DTY+DVA+DSE+DNE)
show sticks, dres and not name C+N+O
color teal, dres

orient chain C
zoom (chain C extend 10), buffer=3
ray 1800,1350
png docs/figures/fig_non_alpha.png, dpi=300
