"use client";

import { useState, useEffect } from "react";
import { getDriftReport, DriftResponse } from "@/lib/api";
import { AlertTriangle, CheckCircle, Activity } from "lucide-react";

export default function DriftPage() {
  const [fileId, setFileId] = useState("");
  const [jsonInput, setJsonInput] = useState("");
  const [result, setResult] = useState<DriftResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setFileId(localStorage.getItem("file_id") || "");
  }, []);

  const handleDrift = async () => {
    setLoading(true);
    setError("");
    try {
      const currentData = JSON.parse(jsonInput);
      const res = await getDriftReport(fileId, currentData);
      setResult(res);
    } catch (e: any) {
      if (e instanceof SyntaxError) {
        setError("Invalid JSON. Please check your input format.");
      } else {
        setError(e.response?.data?.detail || "Drift computation failed");
      }
    }
    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">Drift Monitor</h1>

      {/* Input */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <div className="mb-4">
          <label className="text-sm text-gray-400 mb-1 block">File ID</label>
          <input
            value={fileId}
            onChange={(e) => setFileId(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm font-mono"
          />
        </div>
        <div className="mb-4">
          <label className="text-sm text-gray-400 mb-1 block">
            Current Data (JSON array of feature dicts, min 10 rows)
          </label>
          <textarea
            value={jsonInput}
            onChange={(e) => setJsonInput(e.target.value)}
            placeholder='[{"feature1": 1.0, "feature2": 2.0}, ...]'
            rows={8}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm font-mono"
          />
        </div>
        <button
          onClick={handleDrift}
          disabled={loading || !fileId || !jsonInput}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 py-2 rounded-lg font-medium transition-colors"
        >
          {loading ? "Computing drift..." : "Run Drift Report"}
        </button>
      </div>

      {/* Result */}
      {result && (
        <>
          {/* Status Card */}
          <div className={`rounded-xl p-6 border ${
            result.dataset_drift_detected
              ? "bg-red-900/20 border-red-800"
              : "bg-green-900/20 border-green-800"
          }`}>
            <div className="flex items-center gap-3 mb-3">
              {result.dataset_drift_detected
                ? <AlertTriangle className="text-red-400" size={24} />
                : <CheckCircle className="text-green-400" size={24} />}
              <h2 className="text-lg font-semibold">
                {result.dataset_drift_detected ? "Drift Detected" : "No Drift Detected"}
              </h2>
            </div>
            <p className="text-gray-300 mb-4">{result.message}</p>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div className="bg-gray-900/50 rounded-lg p-3">
                <p className="text-gray-400 text-xs">Drift Score</p>
                <p className="text-2xl font-bold text-blue-400">{result.drift_score}</p>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-3">
                <p className="text-gray-400 text-xs">Features Drifted</p>
                <p className="text-2xl font-bold text-orange-400">{result.n_features_drifted}</p>
              </div>
              <div className="bg-gray-900/50 rounded-lg p-3">
                <p className="text-gray-400 text-xs">Current Rows</p>
                <p className="text-2xl font-bold text-purple-400">{result.current_rows}</p>
              </div>
            </div>
          </div>

          {/* Per Feature Drift */}
          {Object.keys(result.feature_drift).length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Activity size={20} />
                Per Feature Drift
              </h2>
              <div className="space-y-3">
                {Object.entries(result.feature_drift).map(([feature, data]) => (
                  <div key={feature} className="bg-gray-800 rounded-lg p-4">
                    <div className="flex justify-between items-center mb-2">
                      <span className="font-medium">{feature}</span>
                      <span className={`text-xs px-2 py-1 rounded-full ${
                        data.drift_detected
                          ? "bg-red-900/50 text-red-300"
                          : "bg-green-900/50 text-green-300"
                      }`}>
                        {data.drift_detected ? "Drifted" : "Stable"}
                      </span>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="flex-1 bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            data.drift_detected ? "bg-red-500" : "bg-green-500"
                          }`}
                          style={{ width: `${Math.min(data.drift_score * 100, 100)}%` }}
                        />
                      </div>
                      <span className="text-sm text-gray-400 w-16 text-right">
                        {data.drift_score.toFixed(3)}
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">
                      {data.stat_test} | threshold: {data.threshold}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg px-4 py-3 text-red-300">
          {error}
        </div>
      )}
    </div>
  );
}