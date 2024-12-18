echo "Creating Aloha Camera example"
cd external/mujoco_menagerie/aloha

patch -p0 < mjx_aloha.patch
patch -p0 < mjx_filtered_cartesian_actuators.patch
patch -p0 < mjx_scene.patch

mv aloha.xml mjx_aloha.xml
mv filtered_cartesian_actuators.xml mjx_filtered_cartesian_actuators.xml
mv scene.xml mjx_scene.xml

git restore aloha.xml filtered_cartesian_actuators.xml scene.xml

# add a box
cat >mjx_single_cube.xml <<EOF
<mujoco model="aloha with a single cube">
  <size nuserdata="1"/>
  <include file="mjx_scene.xml"/>
  <worldbody>
    <body name="box" pos="0.35 0.2 0.025">
      <freejoint/>
      <geom name="box" type="box" size="0.015 0.02 0.03" condim="3"
        friction="2.5 .03 .003" rgba="0 1 0 1" contype="1" conaffinity="1"/>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="
      0 -0.96 1.16 0 -0.3 0 0.0084 0.0084
      0 -0.96 1.16 0 -0.3 0 0.0084 0.0084
      0.35 0.2 0.025 1 0 0 0"
      act= "-0.1 0 0 0 0 0 0.03 0.1 0 0 0 0 0 0.03"
      ctrl="-0.1 0 0 0 0 0 0.03 0.1 0 0 0 0 0 0.03"
    />
  </keyframe>
</mujoco>
EOF