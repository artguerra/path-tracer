import { vec3 } from "wgpu-matrix";

export interface Mesh {
  positions: Float32Array;
  normals: Float32Array;
  indices: Uint32Array;
}

export interface MeshInstance {
  mesh: Mesh;
  materialIndex: number;
}

export interface MergedGeometry {
  positions: Float32Array<ArrayBuffer>;
  normals: Float32Array<ArrayBuffer>;
  indices: Uint32Array<ArrayBuffer>;
  instances: Uint32Array<ArrayBuffer>; // packed data for WGSL: array<Mesh>
  primitiveMeshIndices: Uint32Array<ArrayBuffer>;
}

export function mergeMeshes(meshInstances: MeshInstance[]): MergedGeometry {
  const numVertices = meshInstances.reduce((acc, cur) => acc + cur.mesh.positions.length, 0);
  const numIndices = meshInstances.reduce((acc, cur) => acc + cur.mesh.indices.length, 0);
  const numTris = numIndices / 3;

  const merged: MergedGeometry = {
    positions: new Float32Array(numVertices),
    normals: new Float32Array(numVertices),
    indices: new Uint32Array(numIndices),
    instances: new Uint32Array(meshInstances.length * 4), // 4 u32s per instance
    primitiveMeshIndices: new Uint32Array(numTris),
  };

  let posOffset = 0;
  let idxOffset = 0;
  let triOffset = 0;

  for (let i = 0; i < meshInstances.length; ++i) {
    const { mesh, materialIndex } = meshInstances[i];
    const triCount = mesh.indices.length / 3;

    merged.positions.set(mesh.positions, posOffset);
    merged.normals.set(mesh.normals, posOffset);
    merged.indices.set(mesh.indices, idxOffset);
    merged.primitiveMeshIndices.fill(i, triOffset, triOffset + triCount);

    const vertexOffset = posOffset / 3;
    const instanceIdx = i * 4;
    merged.instances[instanceIdx + 0] = vertexOffset; // posOffset
    merged.instances[instanceIdx + 1] = triOffset; // triOffset
    merged.instances[instanceIdx + 2] = triCount; // numOfTriangles
    merged.instances[instanceIdx + 3] = materialIndex; // materialIndex

    posOffset += mesh.positions.length;
    idxOffset += mesh.indices.length;
    triOffset += triCount;
  }

  return merged;
}

// --------------------  PRIMITIVES --------------------

export function computeNormals(mesh: Mesh) {
  const numTri = mesh.indices.length / 3;
  const length = mesh.normals.length;

  for (let i = 0; i < length; ++i) {
    mesh.normals[i] = 0.0;
  }

  for (let i = 0; i < numTri; ++i) {
    const v0 = mesh.indices[3 * i];
    const v1 = mesh.indices[3 * i + 1];
    const v2 = mesh.indices[3 * i + 2];

    const p0 = vec3.create(mesh.positions[3 * v0], mesh.positions[3 * v0 + 1], mesh.positions[3 * v0 + 2]);
    const p1 = vec3.create(mesh.positions[3 * v1], mesh.positions[3 * v1 + 1], mesh.positions[3 * v1 + 2]);
    const p2 = vec3.create(mesh.positions[3 * v2], mesh.positions[3 * v2 + 1], mesh.positions[3 * v2 + 2]);

    const e01 = vec3.sub(p1, p0);
    const e12 = vec3.sub(p2, p1);
    const c = vec3.cross(e01, e12);
    const nt = vec3.normalize(c);

    mesh.normals[3 * v0] += nt[0];
    mesh.normals[3 * v0 + 1] += nt[1];
    mesh.normals[3 * v0 + 2] += nt[2];
    mesh.normals[3 * v1] += nt[0];
    mesh.normals[3 * v1 + 1] += nt[1];
    mesh.normals[3 * v1 + 2] += nt[2];
    mesh.normals[3 * v2] += nt[0];
    mesh.normals[3 * v2 + 1] += nt[1];
    mesh.normals[3 * v2 + 2] += nt[2];
  }

  for (let i = 0; i < length / 3; ++i) {
    const ni = vec3.create(mesh.normals[3 * i], mesh.normals[3 * i + 1], mesh.normals[3 * i + 2]);
    const nni = vec3.normalize(ni);
    mesh.normals[3 * i] = nni[0];
    mesh.normals[3 * i + 1] = nni[1];
    mesh.normals[3 * i + 2] = nni[2];
  }
}

export function createQuad(origin: number[], edge0: number[], edge1: number[]): Mesh {
  const a = vec3.add(origin, edge0);
  const b = vec3.add(a, edge1);
  const c = vec3.add(origin, edge1);
  const n = vec3.cross(edge0, edge1);
  const indices = [0, 1, 2, 0, 2, 3];
  
  return {
    positions: new Float32Array([...origin, ...a, ...b, ...c]),
    normals: new Float32Array([...n, ...n, ...n, ...n]),
    indices: new Uint32Array(indices),
  };
}

export function createBox(
  origin: number[], width: number, height: number, length: number, angle: number = 0.0
): Mesh {
  const w = width / 2.0;
  const h = height / 2.0;
  const l = length / 2.0;
  const positions = [
    0.0, 0.0,  2*l,
    2*w, 0.0,  2*l,
    2*w,  2*h,  2*l,
    0.0,  2*h,  2*l,
    0.0, 0.0, 0.0,
    2*w, 0.0, 0.0,
    2*w,  2*h, 0.0,
    0.0,  2*h, 0.0,
  ];

  const den = vec3.length([w, h, l]);
  const wn = w / den;
  const hn = h / den;
  const ln = l / den;

  const normals = [
    -wn, -hn,  ln,
    wn, -hn,  ln,
    wn,  hn,  ln,
    -wn,  hn,  ln,
    -wn, -hn, -ln,
    wn, -hn, -ln,
    wn,  hn, -ln,
    -wn,  hn, -ln,
  ];

  const sinTheta = Math.sin(angle);
  const cosTheta = Math.cos(angle);

  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i];
    const y = positions[i + 1];
    const z = positions[i + 2];

    const rx = x * cosTheta - z * sinTheta;
    const ry = y;
    const rz = x * sinTheta + z * cosTheta;

    positions[i] = rx + origin[0];
    positions[i + 1] = ry + origin[1];
    positions[i + 2] = rz + origin[2];
  }

  for (let i = 0; i < normals.length; i += 3) {
    const nx = normals[i];
    const ny = normals[i + 1];
    const nz = normals[i + 2];

    normals[i] = nx * cosTheta - nz * sinTheta;
    normals[i + 1] = ny;
    normals[i + 2] = nx * sinTheta + nz * cosTheta;
  }

  const indices = [
    0, 1, 2,  0, 2, 3,  1, 5, 6,  1, 6, 2,
    5, 4, 7,  5, 7, 6,  4, 0, 3,  4, 3, 7,
    3, 2, 6,  3, 6, 7,  4, 5, 1,  4, 1, 0,
  ];

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices)
  };
}

export function createSphere(origin: number[], radius: number, latitudeRes: number, longitudeRes: number): Mesh {
  const positions: number[] = [];
  const normals: number[] = [];
  const indices: number[] = [];

  for (let lat = 0; lat <= latitudeRes; lat++) {
    const theta = lat * Math.PI / latitudeRes;
    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    for (let lon = 0; lon <= longitudeRes; lon++) {
      const phi = lon * 2 * Math.PI / longitudeRes;
      const sinPhi = Math.sin(phi);
      const cosPhi = Math.cos(phi);

      const x = cosPhi * sinTheta;
      const y = cosTheta;
      const z = sinPhi * sinTheta;

      positions.push(origin[0] + radius * x, origin[1] + radius * y, origin[2] + radius * z);
      normals.push(x, y, z);
    }
  }

  for (let lat = 0; lat < latitudeRes; lat++) {
    for (let lon = 0; lon < longitudeRes; lon++) {
      const first = lat * (longitudeRes + 1) + lon;
      const second = first + longitudeRes + 1;

      indices.push(first, first + 1, second, second, first + 1, second + 1);
    }
  }

  return {
    positions: new Float32Array(positions),
    normals: new Float32Array(normals),
    indices: new Uint32Array(indices)
  };
}
