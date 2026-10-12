# The Last Supper as a real 3D set

A real 3D world built from the painting, for a camera that flies through it in one continuous shot.
The painting dissolves into the 3D world at the start and comes back at the end. The figures never move.

- **People:** hyper-real recreations of each group in the painting's poses (gpt-image-2.5, using the fresco as reference).
  They are lifted to textured 3D meshes with Tripo H3.1 image-to-3D: Jesus plus the four groups of three apostles, all 13 figures.
- **Room:** `build_set.py`, built procedurally at metre scale.
  It has the coffered ceiling, tapestried side walls, the three back windows opening onto an AI landscape, and the long table with linen, pewter, bread and wine.
- **Light:** a low sun behind the windows throws shafts across the table, with a warm fill, a rim light and AgX High Contrast.
- **Camera:** `sample_shot.py` renders the 7 s sample, resumable frame by frame (Cycles, CPU):
  1. It starts from the painting's own viewpoint and pushes in.
  2. It cranes over Bartholomew, James and Andrew.
  3. It circles behind Judas, Peter and John.
  4. It comes round to find Christ.
- **Finish:** `finish_sample.sh` handles the opening dissolve from the real fresco (aligned to the first frame), plus bloom, haze, grade, grain and the score.
- **Helpers:** `preview_glb.py` renders turntable checks of the meshes, and `wide_check.py` renders establishing views of the set.
