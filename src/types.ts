import type { Vec3 } from "./math";

export type LightType = "point" | "area";

export interface LightSourceBase {
  type: LightType;
  position: Vec3;
  intensity: number;
  color: Vec3;
  rayTracedShadows: number; // 0 or 1
}

export interface PointLight extends LightSourceBase {
  type: "point";
}

export type LightSource = PointLight;

export interface Material {
  albedo: Vec3;
  roughness: number;
  metalness: number;
  materialType: number; // 0 = standard, 1 = procedural, 2 = emissive
  emissionStrength: number; // 0 = not emissive, >0 emissive
}
