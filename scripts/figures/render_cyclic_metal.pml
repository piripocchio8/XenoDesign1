load /home/user/claude_projects/XenoDesign1/XenoDesign1_local_ref/denovo_cyclic_metal/loop/iter_011/chai_out/pred.model_idx_0.cif, cyc

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
set dash_color, grey25
set dash_gap, 0.25
set dash_width, 3.0
set label_size, 26
set label_color, black
set label_outline_color, white
set float_labels, 1
set depth_cue, 0

hide everything
set cartoon_loop_radius, 0.50
set cartoon_tube_radius, 0.50
set stick_radius, 0.22
show cartoon, chain B
color lightblue, chain B
set cartoon_transparency, 0.45, chain B

select his19, (chain B and resi 19 and resn HIS+DHI)
select his23, (chain B and resi 23 and resn HIS+DHI)
show sticks, (his19 or his23) and not name C+N+O
color salmon, (his19 or his23)

select zn, (symbol Zn)
show spheres, zn
color purple, zn
set sphere_scale, 0.40, zn

# Exactly two clean coordination measurements: Zn to each coordinating His ND1.
distance dz1, zn, (his19 and name ND1)
distance dz2, zn, (his23 and name ND1)
color grey25, dz1
color grey25, dz2
# Numeric values would overprint at this scale; the caption states Zn-N ~2.1-2.3 A.
hide labels, dz1
hide labels, dz2

# Tight, central view on the coordination site (a little adjacent backbone for context).
orient (zn or his19 or his23)
zoom (zn or his19 or his23), 5
turn x, -8
turn y, 6

set ray_trace_fog, 0
ray 1800,1350
png /home/user/claude_projects/XenoDesign1/docs/figures/fig_cyclic_metal.png, dpi=300
