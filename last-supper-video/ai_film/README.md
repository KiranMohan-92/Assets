# The Hour Before (AI film cut)

A 40-second original short built on Leonardo's *Last Supper*. The video is in Higgsfield: https://d2ol7oe51mr4n9.cloudfront.net/user_2zHlGJB9WmlHiGuHON7heHIhpuF/dcbc3ff6-6809-434c-81df-4c2f93248567.mp4

1. **Restored keyframe.** GPT Image 2.5 repainted the public-domain Wikimedia scan as a clean, lifelike version.
2. **Shot keyframes.** GPT Image 2.5 made eight new frames from that reference, so faces, robes and the room stay consistent.
3. **Shots.** Kling 3.0 Omni (Pro, 5 s each, 8 s for the finale) animated each frame with its own camera move:

| time | shot |
|---|---|
| 0.0 | macro: the wine trembles, rack focus to Christ's hands |
| 4.6 | dusk drone flight into the glowing window |
| 9.2 | high crane through the window light |
| 13.4 | dolly along the plates to the hand on the money pouch |
| 16.8 | Judas turns into shadow |
| 20.2 | low arc around Thomas, James and Philip |
| 24.0 | Christ lifts his eyes to the camera |
| 28.6 | crane pull-back into the fresco itself, title |

4. **Score.** `epic_score.py` synthesizes an original orchestral score: piano, strings, choir, horns, taiko, timpani and a hall reverb. It is in D minor, resolves to D major, and its cue points match the cuts above.
5. **Edit.** `edit.sh` cuts the shots on the cues, grades them, adds a 2.39:1 letterbox and grain, then adds the title and mixes the score.

Total cost was about 57 Higgsfield credits.

## Vertical 9:16 reel — "Into the Painting" (one continuous shot)

The figures never move; only the camera does. Five Kling 3.0 Omni segments are chained frame to frame, each ending on the keyframe the next one starts from, so the stitched result plays as one unbroken flight: the camera goes down the refectory hall, through the fresco, past the wine and bread, the money pouch and spilled salt, and around Judas, Peter and John. It then swerves past Thomas, James and Philip and lands on Christ's face. A final 10 s segment pulls back slowly to the window halo. The score is the same cue as the 16:9 film, re-cut so the drop hits the moment the camera lands on Christ (24.0 s).

- Assembly: `edit_vertical.sh DIR` (DIR holds v1–v5.mp4, vz.mp4 and score.wav)
- Final (1080×1920, 34 s): https://d2ol7oe51mr4n9.cloudfront.net/user_2zHlGJB9WmlHiGuHON7heHIhpuF/be8e6e17-1a17-44ed-a32e-2f4fe0f6da68.mp4
