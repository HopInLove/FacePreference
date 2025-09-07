import { Storage } from "./storage.js";
import {
  CalibrationEvent,
  CandidateFeatureRow,
  Face,
  RankedCandidate,
  TasteVector,
  Vector,
} from "../types.js";

export class MemoryStorage implements Storage {
  private catalog: Map<string, Face> = new Map();
  private userFaces: Map<string, Vector[]> = new Map();
  private calibration: Map<string, CalibrationEvent[]> = new Map();
  private tastes: Map<string, TasteVector> = new Map();
  private candidates: Map<string, RankedCandidate[]> = new Map();

  seedCatalog(faces: Face[]) {
    faces.forEach((f) => this.catalog.set(f.faceId, f));
  }
  seedUserFace(userId: string, vec: Vector) {
    const arr = this.userFaces.get(userId) ?? [];
    arr.push(vec);
    this.userFaces.set(userId, arr);
  }

  async listCatalogFaces(): Promise<Face[]> {
    return Array.from(this.catalog.values());
  }
  getFaceVector(faceId: string): Vector | null {
    return this.catalog.get(faceId)?.vector ?? null;
  }

  async upsertUserFace(userId: string, vector: Vector): Promise<void> {
    const arr = this.userFaces.get(userId) ?? [];
    arr.push(vector);
    this.userFaces.set(userId, arr);
  }

  async getUserFaceAggregate(userId: string): Promise<Vector | null> {
    const arr = this.userFaces.get(userId);
    if (!arr || arr.length === 0) return null;
    const dim = arr[0].length;
    const mean = new Array(dim).fill(0);
    for (const v of arr) for (let i = 0; i < dim; i++) mean[i] += v[i];
    for (let i = 0; i < dim; i++) mean[i] /= arr.length;
    const norm = Math.sqrt(mean.reduce((s, x) => s + x * x, 0)) || 1;
    return mean.map((x) => x / norm);
  }

  async saveCalibrationEvent(ev: CalibrationEvent): Promise<void> {
    const arr = this.calibration.get(ev.userId) ?? [];
    arr.push(ev);
    this.calibration.set(ev.userId, arr);
  }

  async getCalibrationEvents(userId: string): Promise<CalibrationEvent[]> {
    return this.calibration.get(userId) ?? [];
  }

  async saveTasteVector(tv: TasteVector): Promise<void> {
    this.tastes.set(tv.userId, tv);
  }
  async getTasteVector(userId: string): Promise<TasteVector | null> {
    return this.tastes.get(userId) ?? null;
  }

  async saveCandidateCache(userId: string, items: CandidateFeatureRow[]): Promise<void> {
    const ranked: RankedCandidate[] = items
      .filter((r) => r.mgop !== undefined)
      .sort((a, b) => b.mgop! - a.mgop!)
      .map((r) => ({ userId: r.userJ, mgop: r.mgop!, details: r }));
    this.candidates.set(userId, ranked);
  }

  async topNCandidates(userId: string, n: number): Promise<RankedCandidate[]> {
    return (this.candidates.get(userId) ?? []).slice(0, n);
  }
}
