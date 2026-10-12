#!/usr/bin/env bash
# Download everything the 3D set needs (figure meshes, textures, painting, score) into ./assets
set -euo pipefail
mkdir -p assets/glb assets/img
G=https://d8j0ntlcm91z4.cloudfront.net/user_2zHlGJB9WmlHiGuHON7heHIhpuF
get() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
get $G/hf_20261012_172820_9213d29d-bfcf-4bac-9137-a23bfb7a593a.glb assets/glb/jesus.glb
get $G/hf_20261012_172822_6091e984-f9d5-49b7-bfe7-8fc7be7bb51f.glb assets/glb/g2.glb
get $G/hf_20261012_174845_6dec6794-31d4-4a23-ad9f-e35a9204d417.glb assets/glb/g1.glb
get $G/hf_20261012_174850_fec41bfc-2f44-48f8-ada5-9a8f6078f11a.glb assets/glb/g3.glb
get $G/hf_20261012_174852_53a95da0-7408-4c93-87c5-c2713742a1e4.glb assets/glb/g4.glb
get $G/hf_20261012_172610_3cb82fca-a7fa-41aa-a3c4-c0aefc8d40b4.png assets/img/tapestry.png
get $G/hf_20261012_172642_2675b7d6-dfe1-4033-a056-304ce108bb62.png assets/img/landscape.png
echo "assets ready"
