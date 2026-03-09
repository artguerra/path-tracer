import { Camera } from "./camera";
import { mergeMeshes, type MeshInstance, type MergedGeometry } from "./mesh";
import type { GPUApp, GPUAppBase } from "./renderer";
import type { AreaLight, LightSource, Material, PointLight } from "./types";
import { createGPUBuffer } from "./utils";
import { BVHTree } from "./bvh";

export class Scene {
  camera: Camera;
  instances: MeshInstance[];
  materials: Material[];

  pointLights: PointLight[] = [];
  areaLights: AreaLight[] = [];

  bvh?: BVHTree;
  
  mergedGeometry: MergedGeometry;
  time: number = 0;

  // GPU buffers
  buffersInitialized: boolean = false;
  uniformBuffer?: GPUBuffer;
  posBuffer?: GPUBuffer;
  normBuffer?: GPUBuffer;
  triBuffer?: GPUBuffer;
  instanceBuffer?: GPUBuffer;
  matBuffer?: GPUBuffer;
  pointLightBuffer?: GPUBuffer;
  areaLightBuffer?: GPUBuffer;
  bvhBuffer?: GPUBuffer;
  sortedIndicesBuffer?: GPUBuffer;

  pointLightDataArray?: ArrayBuffer;
  areaLightDataArray?: ArrayBuffer;

  constructor(camera: Camera, instances: MeshInstance[], materials: Material[], lights: LightSource[]) {
    this.camera = camera;
    this.instances = instances;
    this.materials = materials;

    for (const light of lights) {
      if (light.type === "point") this.pointLights.push(light);
      if (light.type === "area") this.areaLights.push(light);
    }

    this.mergedGeometry = mergeMeshes(this.instances);
  }

  computeBVH() {
    this.bvh = new BVHTree(this.mergedGeometry);
    this.bvh.buildRecursive(this.bvh.rootIdx);
  }

  createBuffers(app: GPUAppBase) {
    const storageUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;

    this.posBuffer = createGPUBuffer(app.device, this.mergedGeometry.positions, storageUsage);
    this.normBuffer = createGPUBuffer(app.device, this.mergedGeometry.normals, storageUsage);
    this.triBuffer = createGPUBuffer(app.device, this.mergedGeometry.indices, storageUsage);
    this.instanceBuffer = createGPUBuffer(app.device, this.mergedGeometry.instances, storageUsage);

    if (!this.bvh) throw new Error("BVH was not initialized.");
    this.bvhBuffer = createGPUBuffer(app.device, this.bvh.exportBVH(), storageUsage);
    this.sortedIndicesBuffer = createGPUBuffer(app.device, this.bvh.exportSortedIndices(), storageUsage);

    this.uniformBuffer = app.device.createBuffer({
      size: 92 * 4, // 92 floats in scene struct in shader
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // 8 bytes per material
    const matData = new ArrayBuffer(this.materials.length * 8 * 4);
    const matDataF32 = new Float32Array(matData);
    const matDataU32 = new Uint32Array(matData);
    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      
      matDataF32.set(m.albedo, offset);
      matDataF32[offset + 3] = m.roughness;
      matDataF32[offset + 4] = m.metalness;
      matDataU32[offset + 5] = m.useProceduralTexture ? 1 : 0;
      // 6, 7 are padding
    }
    this.matBuffer = createGPUBuffer(app.device, matData, storageUsage);

    this.packLightData();
    this.pointLightBuffer = createGPUBuffer(app.device, new Uint8Array(this.pointLightDataArray!), storageUsage);
    this.areaLightBuffer = createGPUBuffer(app.device, new Uint8Array(this.areaLightDataArray!), storageUsage);

    this.buffersInitialized = true;
  }

  private packLightData() {
    // point lights (8 floats/u32s = 32 bytes each)
    const numPointLights = Math.max(1, this.pointLights.length); // at least allocate 1
    this.pointLightDataArray = new ArrayBuffer(numPointLights * 8 * 4);
    const pF32View = new Float32Array(this.pointLightDataArray);
    const pU32View = new Uint32Array(this.pointLightDataArray);

    for (let i = 0; i < this.pointLights.length; i++) {
      const l = this.pointLights[i];
      const offset = i * 8;

      pF32View.set(l.position, offset);
      pF32View[offset + 3] = l.intensity;

      pF32View.set(l.color, offset + 4);
      pU32View[offset + 7] = l.rayTracedShadows;
    }

    // area lights (16 floats/u32s = 64 bytes each)
    const numAreaLights = Math.max(1, this.areaLights.length); // at least allocate 1
    this.areaLightDataArray = new ArrayBuffer(numAreaLights * 16 * 4);
    const aF32View = new Float32Array(this.areaLightDataArray);
    const aU32View = new Uint32Array(this.areaLightDataArray);

    for (let i = 0; i < this.areaLights.length; i++) {
      const l = this.areaLights[i];
      const offset = i * 16;

      aF32View.set(l.position, offset);
      aF32View[offset + 3] = l.intensity;

      aF32View.set(l.color, offset + 4);
      aU32View[offset + 7] = l.rayTracedShadows;

      aF32View.set(l.u, offset + 8);
      aF32View[offset + 11] = 0.0; // padding

      aF32View.set(l.v, offset + 12);
      aF32View[offset + 15] = 0.0; // padding

    }
  }

  animate() {
    this.time += 1.0;
  }

  updateMaterials(app: GPUApp) {
    if (!this.buffersInitialized) return;

    const matData = new Float32Array(this.materials.length * 8);

    for (let i = 0; i < this.materials.length; i++) {
      const m = this.materials[i];
      const offset = i * 8;
      matData.set(m.albedo, offset);
      matData[offset + 3] = m.roughness;
      matData[offset + 4] = m.metalness;
    }

    app.device.queue.writeBuffer(this.matBuffer!, 0, matData);
  }

  updateGPU(app: GPUApp) {
    if (!this.buffersInitialized) return;

    const sceneData = new Float32Array(92);

    // matrices
    sceneData.set(this.camera.modelMat, 0);
    sceneData.set(this.camera.viewMat, 16);
    sceneData.set(this.camera.invViewMat, 32);
    sceneData.set(this.camera.transInvModelMat, 48);
    sceneData.set(this.camera.projMat, 64);
    
    // camera
    sceneData[80] = this.camera.fov;
    sceneData[81] = this.camera.aspect;
    
    // scene
    sceneData[84] = app.canvas.width;
    sceneData[85] = app.canvas.height;
    sceneData[86] = this.instances.length;
    sceneData[87] = this.pointLights.length;
    sceneData[88] = this.areaLights.length;
    sceneData[89] = this.time;
    sceneData[90] = 0.0; // padding
    sceneData[91] = 0.0; // padding

    app.device.queue.writeBuffer(this.uniformBuffer!, 0, sceneData);

    // update lights
    // repack and upload it again
    this.packLightData();
    app.device.queue.writeBuffer(this.pointLightBuffer!, 0, this.pointLightDataArray!);
    app.device.queue.writeBuffer(this.areaLightBuffer!, 0, this.areaLightDataArray!);
  }
}
