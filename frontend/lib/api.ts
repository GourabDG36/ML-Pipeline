import axios from "axios";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export const api = axios.create({
  baseURL: API_BASE,
  timeout: 600000, // 10 minutes for training
});

// Types
export interface UploadResponse {
  file_id: string;
  filename: string;
  rows: number;
  columns: number;
  column_names: string[];
  message: string;
}

export interface ValidationResponse {
  file_id: string;
  target_column: string;
  is_valid: boolean;
  errors: string[];
  warnings: string[];
  numerical_features: string[];
  categorical_features: string[];
  dropped_features: string[];
  drop_reasons: Record<string, string>;
  info: Record<string, any>;
}

export interface TrainResponse {
  file_id: string;
  best_model_name: string;
  best_metrics: Record<string, number>;
  all_results: Record<string, any>;
  message: string;
}

export interface PredictResponse {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  shap_values: Record<string, number>;
  shap_error: string | null;
  features_used: string[];
  model_name: string;
}

export interface DriftResponse {
  status: string;
  message: string;
  dataset_drift_detected: boolean;
  drift_score: number;
  n_features_drifted: number;
  feature_drift: Record<string, {
    drift_detected: boolean;
    drift_score: number;
    stat_test: string;
    threshold: number;
  }>;
  reference_rows: number;
  current_rows: number;
}

// API functions
export const uploadCSV = async (file: File): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append("file", file);
  const res = await api.post("/upload", formData);
  return res.data;
};

export const validateCSV = async (
  file_id: string,
  target_column: string
): Promise<ValidationResponse> => {
  const res = await api.post(`/validate?file_id=${file_id}&target_column=${target_column}`);
  return res.data;
};

export const trainModels = async (
  file_id: string,
  target_column: string
): Promise<TrainResponse> => {
  const res = await api.post("/train", { file_id, target_column });
  return res.data;
};

export const predict = async (
  file_id: string,
  features: Record<string, any>
): Promise<PredictResponse> => {
  const res = await api.post("/predict", { file_id, features });
  return res.data;
};

export const getDriftReport = async (
  file_id: string,
  current_data: Record<string, any>[]
): Promise<DriftResponse> => {
  const res = await api.post("/drift-report", { file_id, current_data });
  return res.data;
};