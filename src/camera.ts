import { type Vec3, vec3, type Mat4, mat4 } from "wgpu-matrix";

export class Camera {
  // world vectors
  position: Vec3;
  front: Vec3 = vec3.create(0, 0, -1);
  right: Vec3 = vec3.create(1, 0, 0);
  worldUp: Vec3 = vec3.create(0, 1, 0);

  // matrices
  modelMat: Mat4;
  viewMat: Mat4;
  invViewMat: Mat4;
  transInvModelMat: Mat4;
  projMat: Mat4;

  // camera parameters
  aspect: number;
  fov: number = Math.PI / 4.0;
  near: number = 0.1;
  far: number = 10000.0;

  // camera control
  yaw: number = 0; // start by looking down the -Z axis
  pitch: number = 0;

  rotateSpeed: number = 0.001;
  moveSpeed: number = 0.05;

  dragging: boolean = false;
  lastX: number = 0;
  lastY: number = 0;

  constructor(position: Vec3, aspectRatio: number) {
    this.position = position;
    this.aspect = aspectRatio;

    this.modelMat = mat4.identity();
    this.viewMat  = mat4.identity(); 
    this.invViewMat = mat4.identity();
    this.transInvModelMat = mat4.identity();
    this.projMat = mat4.perspective(
      this.fov,
      this.aspect,
      this.near,
      this.far
    );
    
    this.updateVectors();
  }

  updateVectors() {
    this.front[0] = -Math.cos(this.pitch) * Math.sin(this.yaw);
    this.front[1] = -Math.sin(this.pitch);
    this.front[2] = -Math.cos(this.pitch) * Math.cos(this.yaw);
    vec3.normalize(this.front, this.front);

    vec3.cross(this.front, this.worldUp, this.right);
    vec3.normalize(this.right, this.right);
  }

  updateCamera() {
    this.updateVectors();
    
    const target = vec3.add(this.position, this.front);
    mat4.lookAt(this.position, target, this.worldUp, this.viewMat);
    
    this.invViewMat = mat4.invert(this.viewMat);
    this.transInvModelMat = mat4.transpose(mat4.invert(this.modelMat));
  }

  processMouseMovement(dx: number, dy: number) {
    this.yaw -= dx * this.rotateSpeed;
    this.pitch += dy * this.rotateSpeed;

    const maxPitch = Math.PI / 2 - 0.01;
    this.pitch = Math.max(-maxPitch, Math.min(maxPitch, this.pitch));
  }

  processKeyboard(forward: number, right: number, up: number) {
    if (forward !== 0) vec3.addScaled(this.position, this.front, forward * this.moveSpeed, this.position);
    if (right !== 0) vec3.addScaled(this.position, this.right, right * this.moveSpeed, this.position);
    if (up !== 0) vec3.addScaled(this.position, this.worldUp, up * this.moveSpeed, this.position);
  }
}