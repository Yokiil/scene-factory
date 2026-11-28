/**
 * Precision 3D Studio Viewport (Three.js)
 * Renders MuJoCo XML inline multi-part meshes + primitives with studio 3-point
 * lighting, fading floor grid, interactive selection, 3D OBB collision boxes,
 * separation axis arrows, and nested SE(3) coordinate triads.
 */

import * as THREE from "three";
import { OrbitControls } from "/vendor/OrbitControls.js";

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

function mat3ToThreeMatrix4(pos, rot) {
  const m = new THREE.Matrix4();
  m.set(
    rot[0][0], rot[0][1], rot[0][2], pos[0],
    rot[1][0], rot[1][1], rot[1][2], pos[1],
    rot[2][0], rot[2][1], rot[2][2], pos[2],
    0, 0, 0, 1,
  );
  return m;
}

export class StudioViewport {
  constructor(container, { onSelect } = {}) {
    this.container = container;
    this.onSelect = onSelect || (() => {});
    this.theme = "dark";
    this.layers = { grid: true, obb: true, axes: false };
    this.bodyMeshes = new Map();
    this.bodyGeoms = new Map();
    this.selectedBody = null;
    this.lastBounds = null;
    this.lastAnalysis = null;

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 16 / 10, 0.02, 80);
    this.camera.position.set(2.1, -2.3, 1.75);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.08;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxPolarAngle = Math.PI * 0.495;
    this.controls.minDistance = 0.35;
    this.controls.maxDistance = 24;

    // Studio 3-Point Lighting (Key + Cool Rim + Soft Ambient Fill)
    this.hemi = new THREE.HemisphereLight(0xf1f5f9, 0x1e293b, 0.9);
    this.scene.add(this.hemi);

    this.keyLight = new THREE.DirectionalLight(0xfff7ed, 2.2);
    this.keyLight.position.set(3.2, -3.6, 5.4);
    this.keyLight.castShadow = true;
    this.keyLight.shadow.mapSize.set(2048, 2048);
    this.keyLight.shadow.bias = -0.0004;
    this.keyLight.shadow.camera.near = 0.2;
    this.keyLight.shadow.camera.far = 18;
    this.keyLight.shadow.camera.left = -3.5;
    this.keyLight.shadow.camera.right = 3.5;
    this.keyLight.shadow.camera.top = 3.5;
    this.keyLight.shadow.camera.bottom = -3.5;
    this.scene.add(this.keyLight);

    this.rimLight = new THREE.DirectionalLight(0x7dd3fc, 0.85);
    this.rimLight.position.set(-3.5, 4.0, 2.6);
    this.scene.add(this.rimLight);

    // Groups
    this.floorGroup = new THREE.Group();
    this.worldGroup = new THREE.Group();
    this.obbGroup = new THREE.Group();
    this.axesGroup = new THREE.Group();
    this.selectGroup = new THREE.Group();
    this.scene.add(this.floorGroup, this.worldGroup, this.obbGroup, this.axesGroup, this.selectGroup);

    this._buildFloor();
    this.setTheme("dark");

    // Click-to-select body via raycasting
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    let downPos = null;
    this.renderer.domElement.addEventListener("pointerdown", (e) => {
      downPos = { x: e.clientX, y: e.clientY };
    });
    this.renderer.domElement.addEventListener("pointerup", (e) => {
      if (!downPos || Math.hypot(e.clientX - downPos.x, e.clientY - downPos.y) > 5) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      this.pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const hits = this.raycaster.intersectObjects(this.worldGroup.children, true);
      const hit = hits.find((h) => h.object.userData && h.object.userData.bodyName);
      this.selectBody(hit ? hit.object.userData.bodyName : null);
      this.onSelect(this.selectedBody);
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.container);
    this.resize();
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  _buildFloor() {
    this._clearGroup(this.floorGroup);
    const dark = this.theme === "dark";

    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(6.5, 64),
      new THREE.MeshStandardMaterial({
        color: dark ? 0x10141c : 0xe3e7ef,
        roughness: 0.92,
        metalness: 0.04,
      }),
    );
    disc.position.z = -0.001;
    disc.receiveShadow = true;
    this.floorGroup.add(disc);

    const grid = new THREE.GridHelper(
      8,
      16,
      dark ? 0x263144 : 0xb8c2d1,
      dark ? 0x181f2c : 0xd0d7e2,
    );
    grid.rotation.x = Math.PI / 2;
    grid.position.z = 0.0005;
    grid.material.transparent = true;
    grid.material.opacity = dark ? 0.65 : 0.75;
    this.floorGroup.add(grid);
    this.floorGroup.visible = this.layers.grid;
  }

  setTheme(theme) {
    this.theme = theme === "light" ? "light" : "dark";
    const bg = this.theme === "dark" ? 0x090b10 : 0xeceff4;
    this.scene.background = new THREE.Color(bg);
    this.scene.fog = new THREE.FogExp2(bg, 0.055);
    this.hemi.intensity = this.theme === "dark" ? 0.85 : 1.25;
    this.keyLight.intensity = this.theme === "dark" ? 2.25 : 2.0;
    this._buildFloor();
  }

  setLayer(name, visible) {
    this.layers[name] = visible;
    if (name === "grid") this.floorGroup.visible = visible;
    if (name === "obb") this.obbGroup.visible = visible;
    if (name === "axes") this.axesGroup.visible = visible;
  }

  loadScene(scene3d, analysis) {
    this._clearGroup(this.worldGroup);
    this._clearGroup(this.obbGroup);
    this._clearGroup(this.axesGroup);
    this._clearGroup(this.selectGroup);
    this.bodyMeshes.clear();
    this.bodyGeoms.clear();
    this.selectedBody = null;
    this.lastAnalysis = analysis;

    const meshesMap = scene3d.meshes || {};
    let triCount = 0;

    for (const body of scene3d.bodies || []) {
      const group = new THREE.Group();
      group.name = body.name;

      // Local coordinate triad at body world origin (great for Level 2 nested SE(3) visualization)
      const axes = new THREE.AxesHelper(body.depth > 0 ? 0.14 : 0.22);
      axes.applyMatrix4(mat3ToThreeMatrix4(body.world_pos, body.world_rot));
      this.axesGroup.add(axes);

      for (const g of body.geoms || []) {
        let geometry = null;
        if (g.type === "mesh" && g.mesh && meshesMap[g.mesh]) {
          const mdata = meshesMap[g.mesh];
          geometry = new THREE.BufferGeometry();
          geometry.setAttribute("position", new THREE.Float32BufferAttribute(mdata.vertices, 3));
          if (mdata.faces && mdata.faces.length >= 3) {
            geometry.setIndex(mdata.faces);
            triCount += mdata.faces.length / 3;
          }
          geometry.computeVertexNormals();
        } else if (g.type === "box" && g.size.length >= 3) {
          geometry = new THREE.BoxGeometry(g.size[0] * 2, g.size[1] * 2, g.size[2] * 2);
          triCount += 12;
        } else if (g.type === "sphere" && g.size.length >= 1) {
          geometry = new THREE.SphereGeometry(g.size[0], 24, 18);
          triCount += 720;
        } else if (g.type === "cylinder" && g.size.length >= 2) {
          geometry = new THREE.CylinderGeometry(g.size[0], g.size[0], g.size[1] * 2, 24);
          geometry.rotateX(Math.PI / 2); // Align cylinder along MuJoCo Z-axis
          triCount += 96;
        } else if (g.type === "capsule" && g.size.length >= 2) {
          geometry = new THREE.CapsuleGeometry(g.size[0], g.size[1] * 2, 8, 18);
          geometry.rotateX(Math.PI / 2);
          triCount += 160;
        }
        if (!geometry) continue;

        const [r, gr, b, a] = g.rgba || [0.65, 0.7, 0.76, 1.0];
        const color = new THREE.Color().setRGB(r, gr, b);
        const mat = new THREE.MeshStandardMaterial({
          color,
          roughness: body.dynamic ? 0.36 : 0.66,
          metalness: body.dynamic ? 0.08 : 0.04,
          transparent: a < 0.99,
          opacity: a,
          side: g.type === "mesh" ? THREE.DoubleSide : THREE.FrontSide,
        });
        const mesh = new THREE.Mesh(geometry, mat);
        mesh.applyMatrix4(mat3ToThreeMatrix4(g.pos, g.rot));
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        mesh.userData = { bodyName: body.name };
        group.add(mesh);
      }

      this.worldGroup.add(group);
      this.bodyMeshes.set(body.name, group);
      this.bodyGeoms.set(body.name, body.geoms || []);
    }

    this._buildDiagnosticOverlays(analysis);
    this.obbGroup.visible = this.layers.obb;
    this.axesGroup.visible = this.layers.axes;

    const box = new THREE.Box3().setFromObject(this.worldGroup);
    this.lastBounds = box.isEmpty() ? null : box;
    this.setCameraPreset("persp");
    return { bodies: (scene3d.bodies || []).length, triangles: Math.round(triCount) };
  }

  _buildDiagnosticOverlays(analysis) {
    if (!analysis || !analysis.bodies) return;
    const collidedNames = new Set();
    for (const pair of analysis.colliding_pairs || []) {
      collidedNames.add(pair.body_a);
      collidedNames.add(pair.body_b);
      const ba = analysis.bodies[pair.body_a];
      const bb = analysis.bodies[pair.body_b];
      if (ba && bb && pair.separation_axis) {
        const mid = new THREE.Vector3(
          (ba.world_center_m[0] + bb.world_center_m[0]) / 2,
          (ba.world_center_m[1] + bb.world_center_m[1]) / 2,
          (ba.world_center_m[2] + bb.world_center_m[2]) / 2,
        );
        const dir = new THREE.Vector3(...pair.separation_axis).normalize();
        const arrow = new THREE.ArrowHelper(dir, mid, 0.22, 0xf43f5e, 0.055, 0.032);
        this.obbGroup.add(arrow);
      }
    }

    for (const name of collidedNames) {
      const b = analysis.bodies[name];
      if (b) this._addBodyOverlay(this.obbGroup, name, b, 0xf43f5e, 0.16);
    }

    for (const flt of analysis.floating_bodies || []) {
      const b = analysis.bodies[flt.body];
      if (!b) continue;
      this._addBodyOverlay(this.obbGroup, flt.body, b, 0xf59e0b, 0.14);
      if (flt.clearance_m > 0) {
        const cx = b.world_center_m[0];
        const cy = b.world_center_m[1];
        const zTop = b.aabb_min_m[2];
        const zBot = Math.max(0, zTop - flt.clearance_m);
        const lineGeo = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(cx, cy, zTop),
          new THREE.Vector3(cx, cy, zBot),
        ]);
        const lineMat = new THREE.LineDashedMaterial({ color: 0xf59e0b, dashSize: 0.018, gapSize: 0.012 });
        const line = new THREE.Line(lineGeo, lineMat);
        line.computeLineDistances();
        this.obbGroup.add(line);
      }
    }
  }

  _addBodyOverlay(targetGroup, bodyName, bodyInfo, hexColor, fillOpacity) {
    const geoms = this.bodyGeoms.get(bodyName) || [];
    const primitiveBoxes = geoms.filter((g) => g.type === "box" && g.size && g.size.length >= 3);
    if (primitiveBoxes.length > 0 && primitiveBoxes.length === geoms.length) {
      for (const g of primitiveBoxes) {
        const boxGeo = new THREE.BoxGeometry(g.size[0] * 2 + 0.005, g.size[1] * 2 + 0.005, g.size[2] * 2 + 0.005);
        const mat4 = mat3ToThreeMatrix4(g.pos, g.rot);
        if (fillOpacity > 0) {
          const fill = new THREE.Mesh(
            boxGeo,
            new THREE.MeshBasicMaterial({ color: hexColor, transparent: true, opacity: fillOpacity, depthWrite: false }),
          );
          fill.applyMatrix4(mat4);
          targetGroup.add(fill);
        }
        const edges = new THREE.LineSegments(
          new THREE.EdgesGeometry(boxGeo),
          new THREE.LineBasicMaterial({ color: hexColor }),
        );
        edges.applyMatrix4(mat4);
        targetGroup.add(edges);
      }
      return;
    }
    this._addAabbOverlay(targetGroup, bodyInfo, hexColor, fillOpacity);
  }

  _addAabbOverlay(targetGroup, bodyInfo, hexColor, fillOpacity) {
    const sx = Math.max(bodyInfo.aabb_half_extents_m[0] * 2, 0.01) + 0.006;
    const sy = Math.max(bodyInfo.aabb_half_extents_m[1] * 2, 0.01) + 0.006;
    const sz = Math.max(bodyInfo.aabb_half_extents_m[2] * 2, 0.01) + 0.006;
    const boxGeo = new THREE.BoxGeometry(sx, sy, sz);
    const [cx, cy, cz] = bodyInfo.world_center_m;

    if (fillOpacity > 0) {
      const fill = new THREE.Mesh(
        boxGeo,
        new THREE.MeshBasicMaterial({ color: hexColor, transparent: true, opacity: fillOpacity, depthWrite: false }),
      );
      fill.position.set(cx, cy, cz);
      targetGroup.add(fill);
    }
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(boxGeo),
      new THREE.LineBasicMaterial({ color: hexColor }),
    );
    edges.position.set(cx, cy, cz);
    targetGroup.add(edges);
  }

  selectBody(name) {
    this.selectedBody = name;
    this._clearGroup(this.selectGroup);
    if (!name || !this.lastAnalysis || !this.lastAnalysis.bodies[name]) return;
    this._addBodyOverlay(this.selectGroup, name, this.lastAnalysis.bodies[name], 0x38bdf8, 0.08);
  }

  focusBody(name) {
    this.selectBody(name);
    if (!name || !this.lastAnalysis || !this.lastAnalysis.bodies[name]) return;
    const c = this.lastAnalysis.bodies[name].world_center_m;
    this.controls.target.set(c[0], c[1], c[2]);
  }

  setCameraPreset(mode) {
    const center = new THREE.Vector3(0, 0, 0.45);
    let radius = 1.4;
    if (this.lastBounds) {
      this.lastBounds.getCenter(center);
      const sphere = new THREE.Sphere();
      this.lastBounds.getBoundingSphere(sphere);
      radius = Math.max(sphere.radius, 0.55);
    }
    const dist = radius * 2.55;
    if (mode === "top") {
      this.camera.position.set(center.x, center.y - 0.01, center.z + dist * 1.15);
      this.controls.target.copy(center);
    } else if (mode === "front") {
      this.camera.position.set(center.x, center.y - dist * 1.15, center.z + radius * 0.35);
      this.controls.target.set(center.x, center.y, center.z - radius * 0.12);
    } else {
      const target = new THREE.Vector3(center.x, center.y, center.z - radius * 0.12);
      this.camera.position.set(target.x + dist * 0.72, target.y - dist * 0.88, target.z + dist * 0.62);
      this.controls.target.copy(target);
    }
    this.controls.update();
  }

  resize() {
    const w = Math.max(1, this.container.clientWidth);
    const h = Math.max(1, this.container.clientHeight);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h, false);
  }

  _clearGroup(group) {
    for (const child of [...group.children]) {
      group.remove(child);
      child.traverse((n) => {
        if (n.geometry) n.geometry.dispose();
        if (n.material) {
          if (Array.isArray(n.material)) n.material.forEach((m) => m.dispose());
          else n.material.dispose();
        }
      });
    }
  }
}
