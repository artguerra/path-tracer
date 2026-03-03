import { vec3, type Vec3 } from "wgpu-matrix";
import type { MergedGeometry } from "./mesh";

const MIN_VALUE = -1e30;
const MAX_VALUE = 1e30;

export class AABB {
  minCorner: Vec3;
  maxCorner: Vec3;

  constructor(corner1?: Vec3, corner2?: Vec3) {
    if (!corner1 || !corner2) {
      this.minCorner = vec3.create(MAX_VALUE, MAX_VALUE, MAX_VALUE);
      this.maxCorner = vec3.create(MIN_VALUE, MIN_VALUE, MIN_VALUE);
    } else {
      this.minCorner = vec3.min(corner1, corner2);
      this.maxCorner = vec3.max(corner1, corner2);
    }
  }

  grow(point: Vec3) {
    this.minCorner = vec3.min(this.minCorner, point);
    this.maxCorner = vec3.max(this.maxCorner, point);
  }

  area(): number {
    const diag = vec3.sub(this.maxCorner, this.minCorner);
    return diag[0] * diag[1] + diag[1] * diag[2] + diag[2] * diag[0];
  }

  largestAxis(): number {
    const diag = vec3.sub(this.maxCorner, this.minCorner);

    let largest = 0;
    if (diag[1] > diag[largest]) largest = 1;
    if (diag[2] > diag[largest]) largest = 2;

    return largest;
  }

  exportAABB(): Float32Array<ArrayBuffer> {
    return new Float32Array([...this.minCorner, ...this.maxCorner]);
  }

  static fromVertices(verts: Float32Array): AABB {
    let min = vec3.create(MAX_VALUE, MAX_VALUE, MAX_VALUE);
    let max = vec3.create(MIN_VALUE, MIN_VALUE, MIN_VALUE);

    for (let i = 0; i + 2 < verts.length; i += 3) {
      const p = vec3.create(verts[i], verts[i + 1], verts[i + 2]);

      min = vec3.min(min, p);
      max = vec3.max(max, p);
    }

    return new AABB(min, max);
  }
}

class BVHPrimitive {
  index: number; // index in triangle indices array
  meshIdx: number;
  v0: Vec3;
  v1: Vec3;
  v2: Vec3;
  centroid: Vec3;

  constructor(index: number, meshIdx: number, v0: Vec3, v1: Vec3, v2: Vec3) {
    this.index = index;
    this.meshIdx = meshIdx;
    this.v0 = v0;
    this.v1 = v1;
    this.v2 = v2;
    this.centroid = vec3.divScalar(vec3.add(v0, vec3.add(v1, v2)), 3);
  }
}

class BVHNode {
  bounds: AABB | null; // null if not calculated yet
  leftChild: number | null; // null if leaf
  rightChild: number | null; // null if leaf

  numPrimitives: number | null; // null if interior
  firstPrimitiveOffset: number | null; // null if interior
  splitAxis: number | null; // null if leaf

  constructor(
    left: number | null, right: number | null,
    numPrimitives: number | null, primitiveOffset: number | null,
    splitAxis: number | null
  ) {
    this.bounds = null;
    this.leftChild = left;
    this.rightChild = right;
    this.numPrimitives = numPrimitives;
    this.firstPrimitiveOffset = primitiveOffset;
    this.splitAxis = splitAxis;
  }
}

type BVHHeuristic = "MIDPOINT" | "SAH";

export class BVHTree {
  readonly rootIdx: number = 0;
  heuristic: BVHHeuristic;
  nodes: BVHNode[];
  size: number;

  indices: Uint32Array;
  primitives: BVHPrimitive[];

  constructor(mesh: MergedGeometry, heuristic: BVHHeuristic = "SAH") {
    this.size = 0;
    this.heuristic = heuristic;
    this.indices = mesh.indices;

    const numTris = mesh.indices.length / 3;
    this.primitives = new Array(numTris);
    this.nodes = new Array(2 * numTris - 1);

    for (let i = 0; i < numTris; ++i) {
      const meshIdx = mesh.primitiveMeshIndices[i];
      const baseIdx = 3 * i;
      const vertexOffset = mesh.instances[meshIdx * 4]; // mesh vertex offset

      const i0 = (mesh.indices[baseIdx] + vertexOffset) * 3;
      const i1 = (mesh.indices[baseIdx + 1] + vertexOffset) * 3;
      const i2 = (mesh.indices[baseIdx + 2] + vertexOffset) * 3;

      const v0 = vec3.create(mesh.positions[i0], mesh.positions[i0 + 1], mesh.positions[i0 + 2]);
      const v1 = vec3.create(mesh.positions[i1], mesh.positions[i1 + 1], mesh.positions[i1 + 2]);
      const v2 = vec3.create(mesh.positions[i2], mesh.positions[i2 + 1], mesh.positions[i2 + 2]);

      this.primitives[i] = new BVHPrimitive(baseIdx, meshIdx, v0, v1, v2);
    }

    this.createLeaf(numTris, 0);
    this.updateBounds(this.rootIdx);
    this.size = 1;
  }

  buildRecursive(nodeIdx: number) {
    const node = this.nodes[nodeIdx];
    if (this.heuristic === "MIDPOINT" && node.numPrimitives! <= 2) return;

    let splitAxis = 0;
    let split = 0;
    if (this.heuristic === "SAH") {
      let minCost = MAX_VALUE;
      for (let axis = 0; axis < 3; ++axis) {
        for (let i = 0; i < node.numPrimitives!; ++i) {
          const tri = node.firstPrimitiveOffset! + i;
          const centroid = this.primitives[tri].centroid;
          const cost = this.evalSAH(nodeIdx, centroid, axis);

          if (cost < minCost) {
            minCost = cost;
            splitAxis = axis;
            split = centroid[axis];
          }
        }
      }

      const parentCost = node.numPrimitives! * node.bounds!.area();
      if (minCost >= parentCost) return;
    } else if (this.heuristic === "MIDPOINT") {
      splitAxis = node.bounds!.largestAxis();
      split = vec3.midpoint(node.bounds!.minCorner, node.bounds!.maxCorner)[splitAxis];
    }

    let i = node.firstPrimitiveOffset!;
    let j = i + node.numPrimitives! - 1;
    while (i <= j) {
      if (this.primitives[i].centroid[splitAxis] >= split) {
        const temp = this.primitives[i];
        this.primitives[i] = this.primitives[j];
        this.primitives[j] = temp;
        j--;
      } else {
        i++;
      }
    }

    let nLeft = i - node.firstPrimitiveOffset!;
    if (nLeft === 0 || nLeft === node.numPrimitives!) {
      const first = node.firstPrimitiveOffset!;
      const count = node.numPrimitives!;
      const last = first + count;
      
      const subArray = this.primitives.slice(first, last);
      subArray.sort((a, b) => a.centroid[splitAxis] - b.centroid[splitAxis]);
      
      for (let k = 0; k < count; k++) {
        this.primitives[first + k] = subArray[k];
      }
      
      nLeft = Math.floor(count / 2);
      i = first + nLeft;
    }

    const left = this.createLeaf(nLeft, node.firstPrimitiveOffset!);
    const right = this.createLeaf(node.numPrimitives! - nLeft, i);

    node.numPrimitives = null;
    node.firstPrimitiveOffset = null;
    node.leftChild = left;
    node.rightChild = right;

    this.updateBounds(left);
    this.updateBounds(right);

    this.buildRecursive(left);
    this.buildRecursive(right);
  }

  private evalSAH(nodeIdx: number, centroid: Vec3, axis: number): number {
    const node = this.nodes[nodeIdx];
    const thresh = centroid[axis];
    const leftBox = new AABB(), rightBox = new AABB();

    let leftCount = 0, rightCount = 0;
    for (let i = 0; i < node.numPrimitives!; ++i) {
      const tri = node.firstPrimitiveOffset! + i;
      const prim = this.primitives[tri];

      if (prim.centroid[axis] <= thresh) {
        leftCount++;
        leftBox.grow(prim.v0);
        leftBox.grow(prim.v1);
        leftBox.grow(prim.v2);
      } else {
        rightCount++;
        rightBox.grow(prim.v0);
        rightBox.grow(prim.v1);
        rightBox.grow(prim.v2);
      }
    }

    const cost = leftCount * leftBox.area() + rightCount * rightBox.area();
    return (cost > 0) ? cost : MAX_VALUE;
  }

  private updateBounds(nodeId: number) {
    if (this.nodes[nodeId].numPrimitives === null) throw new Error("Cannot update interior bounds.");

    const first = this.nodes[nodeId].firstPrimitiveOffset!;
    const last = first + this.nodes[nodeId].numPrimitives!;
    const bound = new AABB();

    for (let i = first; i < last; ++i) {
      bound.grow(this.primitives[i].v0);
      bound.grow(this.primitives[i].v1);
      bound.grow(this.primitives[i].v2);
    }

    this.nodes[nodeId].bounds = bound;
  }

  private createLeaf(numPrimitives: number, primitiveOffset: number): number {
    this.nodes[this.size] = new BVHNode(null, null, numPrimitives, primitiveOffset, null);

    return this.size++;
  }

  exportBVH(): ArrayBuffer {
    const bytesPerNode = 8;
    const buffer = new ArrayBuffer(this.size * bytesPerNode * 4);
    const f32View = new Float32Array(buffer);
    const u32View = new Uint32Array(buffer);

    for (let i = 0; i < this.size; ++i) {
      const node = this.nodes[i];
      const idx = i * bytesPerNode;

      f32View.set(node.bounds!.minCorner, idx);
      u32View[idx + 3] = node.leftChild !== null ? node.leftChild : node.firstPrimitiveOffset!;
      f32View.set(node.bounds!.maxCorner, idx + 4);
      u32View[idx + 7] = node.numPrimitives !== null ? node.numPrimitives : 0;
    }

    return buffer;
  }

  exportSortedIndices(): Uint32Array<ArrayBuffer> {
    const sorted = new Uint32Array(this.primitives.length * 4);

    for (let i = 0; i < this.primitives.length; ++i) {
      const originalBase = this.primitives[i].index;
      sorted[4 * i] = this.indices[originalBase];
      sorted[4 * i + 1] = this.indices[originalBase + 1];
      sorted[4 * i + 2] = this.indices[originalBase + 2];
      sorted[4 * i + 3] = this.primitives[i].meshIdx;
    }
    
    return sorted;
  }
}