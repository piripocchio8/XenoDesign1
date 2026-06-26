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

# --- figure: de-novo all-D alpha-helical binder ---
load /home/user/claude_projects/XenoDesign1/XenoDesign1_local_ref/denovo_alpha2/loop/iter_011/chai_out/pred.model_idx_0.cif, obj
hide everything

show cartoon
color grey80, chain A
set cartoon_transparency, 0.45, chain A
color salmon, chain B

select dres, (chain B and resn DAL+DAR+DSG+DAS+DCY+DGN+DGL+DHI+DIL+DLE+DLY+MED+DPN+DPR+DSN+DTH+DTR+DTY+DVA+DSE+DNE)
show sticks, dres and not name C+N+O
color teal, dres
set stick_radius, 0.18, dres

orient chain B
zoom (chain B extend 8), buffer=4
turn y, 15

set ray_trace_mode, 1
ray 1800, 1350
png /home/user/claude_projects/XenoDesign1/docs/figures/fig_alpha.png, dpi=300
