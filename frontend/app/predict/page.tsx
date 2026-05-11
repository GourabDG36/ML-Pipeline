"use client";

import { useState, useEffect } from "react";
import { predict, PredictResponse } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
import { Brain, AlertCircle } from "lucide-react";

export default function PredictPage() {
  const [fileId, setFileId] = useState("");
  const [features, setFeatures] = useState<string[]>([]);
  const [values, setValues] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const savedFileId = localStorage.getItem("file_id") || "";
    const savedFeatures = JSON.parse(localStorage.getItem("features") || "[]");
    setFileId(savedFileId);
    setFeatures(savedFeatures);
    const initValues: Record<string, string> = {};
    savedFeatures.forEach((f: string) => (initValues[f] = ""));
    setValues(initValues);
  }, []);

  const handlePredict = async () => {
    if (!fileId) return;
    setLoading(true);
    setError("");
    try {
      const featureValues: Record<string, any> = {};
      Object.entries(values).forEach(([k, v]) => {
        featureValues[k] = isNaN(Number(v)) ? v : Number(v);
      });
      const res = await predict(fileId, featureValues);
      setResult(res);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Prediction failed");
    }
    setLoading(false);
  };

  const shapData = result
    ? Object.entries(result.shap_values)
        .map(([name, value]) => ({
          name: name.replace("numerical__", "").replace("categorical__", ""),
          importance: parseFloat(value.toFixed(4)),
        }))
        .sort((a, b) => b.importance - a.importance)
    : [];

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">Predict</h1>

      {/* Input Form */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <div className="mb-4">
          <label className="text-sm text-gray-400 mb-1 block">File ID</label>
          <input
            value={fileId}
            onChange={(e) => setFileId(e.target.value)}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm font-mono"
          />
        </div>

        {features.length > 0 && (
          <>
            <h3 className="text-sm text-gray-400 mb-3">Feature Values</h3>
            <div className="grid grid-cols-2 gap-3 mb-4">
              {features.map((f) => (
                <div key={f}>
                  <label className="text-xs text-gray-500 mb-1 block">{f}</label>
                  <input
                    value={values[f] || ""}
                    onChange={(e) => setValues({ ...values, [f]: e.target.value })}
                    placeholder="Enter value"
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm"
                  />
                </div>
              ))}
            </div>
          </>
        )}

        <button
          onClick={handlePredict}
          disabled={loading || !fileId}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 py-2 rounded-lg font-medium transition-colors"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </div>

      {/* Result */}
      {result && (
        <>
          {/* Prediction Card */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <div className="flex items-center gap-3 mb-4">
              <Brain className="text-blue-400" size={24} />
              <h2 className="text-lg font-semibold">Prediction Result</h2>
            </div>
            <div className="grid grid-cols-2 gap-6">
              <div className="text-center bg-blue-900/30 rounded-xl p-6">
                <p className="text-gray-400 text-sm mb-2">Predicted Class</p>
                <p className="text-4xl font-bold text-blue-400">{result.prediction}</p>
                <p className="text-gray-400 text-sm mt-2">
                  Confidence: <span className="text-green-400 font-bold">{(result.confidence * 100).toFixed(1)}%</span>
                </p>
              </div>
              <div>
                <p className="text-gray-400 text-sm mb-3">Class Probabilities</p>
                {Object.entries(result.probabilities).map(([cls, prob]) => (
                  <div key={cls} className="mb-2">
                    <div className="flex justify-between text-sm mb-1">
                      <span>{cls}</span>
                      <span className="text-gray-400">{(prob * 100).toFixed(1)}%</span>
                    </div>
                    <div className="bg-gray-800 rounded-full h-2">
                      <div
                        className="bg-blue-500 h-2 rounded-full transition-all"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* SHAP Chart */}
          {shapData.length > 0 && (
            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
              <h2 className="text-lg font-semibold mb-2">SHAP Feature Importance</h2>
              <p className="text-gray-400 text-sm mb-4">
                Higher value = stronger influence on this prediction
              </p>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={shapData} layout="vertical">
                  <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                  <YAxis
                    type="category"
                    dataKey="name"
                    tick={{ fill: "#9ca3af", fontSize: 11 }}
                    width={140}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                  />
                  <Bar dataKey="importance" fill="#8b5cf6" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-800 rounded-lg px-4 py-3 text-red-300 flex items-center gap-2">
          <AlertCircle size={16} />
          {typeof error === "string" ? error : JSON.stringify(error)}
        </div>
      )}
    </div>
  );
}