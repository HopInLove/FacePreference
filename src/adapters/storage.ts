// src/adapters/storage.ts
import {
  Face,
  CalibrationEvent,
  CandidateFeatureRow,
  RankedCandidate,
  Vector,
  TasteVector,
} from "../types.js";

export interface Storage {
  // Catalog faces
  listCatalogFaces(): Promise<Face[]>;
  getFaceVector(faceId: string): Vector | null;

  // User face embeddings (aggregate robust mean, etc.)
  upsertUserFace(userId: string, vector: Vector): Promise<void>;
  getUserFaceAggregate(userId: string): Promise<Vector | null>;

  // Calibration events
  saveCalibrationEvent(ev: CalibrationEvent): Promise<void>;
  getCalibrationEvents(userId: string): Promise<CalibrationEvent[]>;

  // Learned taste vectors
  saveTasteVector(tv: TasteVector): Promise<void>;
  getTasteVector(userId: string): Promise<TasteVector | null>;

  // Convenience (raw vector attach/read for mutual matcher & UI)
  attachTasteToUser(userId: string, taste: Vector): void;
  getUserTaste(userId: string): Vector | null;

  // Candidate caching (for shortlist & MGOP phase)
  saveCandidateCache(userId: string, items: CandidateFeatureRow[]): Promise<void>;
  topNCandidates(userId: string, n: number): Promise<RankedCandidate[]>;
}
