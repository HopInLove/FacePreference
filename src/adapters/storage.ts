import {
  Face,
  CalibrationEvent,
  CandidateFeatureRow,
  RankedCandidate,
  Vector,
  TasteVector,
} from "../types.js";

export interface Storage {
  listCatalogFaces(): Promise<Face[]>;
  getFaceVector(faceId: string): Vector | null;

  upsertUserFace(userId: string, vector: Vector): Promise<void>;
  getUserFaceAggregate(userId: string): Promise<Vector | null>;

  saveCalibrationEvent(ev: CalibrationEvent): Promise<void>;
  getCalibrationEvents(userId: string): Promise<CalibrationEvent[]>;

  saveTasteVector(tv: TasteVector): Promise<void>;
  getTasteVector(userId: string): Promise<TasteVector | null>;

  saveCandidateCache(userId: string, items: CandidateFeatureRow[]): Promise<void>;
  topNCandidates(userId: string, n: number): Promise<RankedCandidate[]>;
}
