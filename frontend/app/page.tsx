"use client";

import { useState } from "react";
import { uploadCSV, validateCSV, UploadResponse, ValidationResponse } from "@/lib/api";
import { Upload, CheckCircle, AlertCircle, Info } from "lucide-react";

export default function UploadPage() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null);
  const [targetColumn, setTargetColumn] = useState("");
  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleUpload = async () => {
    if (!file) return;
    setLoading(true);
    setError("");
    try {
      const result = await uploadCSV(file);
      setUploadResult(result);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Upload failed");
    }
    setLoading(false);
  };

  const handleValidate = async () => {
    if (!uploadResult || !targetColumn) return;
    setLoading(true);
    setError("");
    try {
      const result = await validateCSV(uploadResult.file_id, targetColumn);
      setValidation(result);
      // Save to localStorage for other pages
      localStorage.setItem("file_id", uploadResult.file_id);
      localStorage.setItem("target_column", targetColumn);
      localStorage.setItem("features", JSON.stringify([
        ...result.numerical_features,
        ...result.categorical_features
      ]));
    } catch (e: any) {
      setError(e.response?.data?.detail || "Validation failed");
    }
    setLoading(false);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">Upload Dataset</h1>

      {/* File Upload */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">1. Select CSV File</h2>
        <div
          className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
          onClick={() => document.getElementById("fileInput")?.click()}
        >
          <Upload className="mx-auto mb-3 text-gray-500" size={40} />
          <p className="text-gray-400">
            {file ? file.name : "Click to select a CSV file"}
          </p>
          <input
            id="fileInput"
            type="file"
            accept=".csv"
            className="hidden"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
          />
        </div>
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="mt-4 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed py-2 rounded-lg font-medium transition-colors"
        >
          {loading ? "Uploading..." : "Upload"}
        </button>
      </div>

      {/* Upload Result */}
      {uploadResult && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h2 className="text-lg font-semibold mb-4">2. Select Target Column</h2>
          <div className="grid grid-cols-3 gap-4 mb-4 text-sm">
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <p className="text-gray-400">Rows</p>
              <p className="text-xl font-bold text-blue-400">{uploadResult.rows}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <p className="text-gray-400">Columns</p>
              <p className="text-xl font-bold text-blue-400">{uploadResult.columns}</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-3 text-center">
              <p className="text-gray-400">File ID</p>
              <p className="text-xs font-mono text-green-400 truncate">{uploadResult.file_id.slice(0, 8)}...</p>
            </div>
          </div>
          <select
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 mb-4"
          >
            <option value="">Select target column...</option>
            {uploadResult.column_names.map((col) => (
              <option key={col} value={col}>{col}</option>
            ))}
          </select>
          <button
            onClick={handleValidate}
            disabled={!targetColumn || loading}
            className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:cursor-not-allowed py-2 rounded-lg font-medium transition-colors"
          >
            {loading ? "Validating..." : "Validate Dataset"}
          </button>
        </div>
      )}

      {/* Validation Result */}
      {validation && (
        <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            {validation.is_valid
              ? <CheckCircle className="text-green-400" size={20} />
              : <AlertCircle className="text-red-400" size={20} />}
            Validation {validation.is_valid ? "Passed" : "Failed"}
          </h2>

          {validation.errors.length > 0 && (
            <div className="mb-4 space-y-2">
              {validation.errors.map((e, i) => (
                <div key={i} className="bg-red-900/30 border border-red-800 rounded-lg px-4 py-2 text-red-300 text-sm">
                  {e}
                </div>
              ))}
            </div>
          )}

          {validation.warnings.length > 0 && (
            <div className="mb-4 space-y-2">
              {validation.warnings.map((w, i) => (
                <div key={i} className="bg-yellow-900/30 border border-yellow-800 rounded-lg px-4 py-2 text-yellow-300 text-sm">
                  {w}
                </div>
              ))}
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-gray-400 mb-2">Numerical Features</p>
              {validation.numerical_features.map((f) => (
                <span key={f} className="inline-block bg-blue-900/50 text-blue-300 px-2 py-1 rounded mr-1 mb-1 text-xs">{f}</span>
              ))}
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <p className="text-gray-400 mb-2">Categorical Features</p>
              {validation.categorical_features.length > 0
                ? validation.categorical_features.map((f) => (
                    <span key={f} className="inline-block bg-purple-900/50 text-purple-300 px-2 py-1 rounded mr-1 mb-1 text-xs">{f}</span>
                  ))
                : <span className="text-gray-500 text-xs">None detected</span>
              }
            </div>
          </div>

          {validation.is_valid && (
            <div className="mt-4 bg-green-900/20 border border-green-800 rounded-lg px-4 py-3 text-green-300 text-sm flex items-center gap-2">
              <Info size={16} />
              Dataset is valid. Go to the Train page to start training.
            </div>
          )}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg px-4 py-3 text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}