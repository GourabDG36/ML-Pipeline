"use client";

import { useState, useEffect } from "react";
import { trainModels, TrainResponse } from "@/lib/api";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Trophy, Clock, Zap } from "lucide-react";

export default function TrainPage() {
  const [fileId, setFileId] = useState("");
  const [targetColumn, setTargetColumn] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TrainResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setFileId(localStorage.getItem("file_id") || "");
    setTargetColumn(localStorage.getItem("target_column") || "");
  }, []);

  const handleTrain = async () => {
    if (!fileId || !targetColumn) return;
    setLoading(true);
    setError("");
    try {
      const res = await trainModels(fileId, targetColumn);
      setResult(res);
      localStorage.setItem("file_id", fileId);
    } catch (e: any) {
      setError(e.response?.data?.detail || "Training failed");
    }
    setLoading(false);
  };

  // Build chart data from results
  const chartData = result
    ? Object.entries(result.all_results).map(([name, data]: [string, any]) => ({
        name: name.replace("_", " "),
        F1: parseFloat((data.metrics.f1 * 100).toFixed(1)),
        Accuracy: parseFloat((data.metrics.accuracy * 100).toFixed(1)),
        "ROC-AUC": parseFloat((data.metrics.roc_auc * 100).toFixed(1)),
      }))
    : [];

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      <h1 className="text-3xl font-bold text-blue-400">Train Models</h1>

      {/* Config */}
      <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-sm text-gray-400 mb-1 block">File ID</label>
            <input
              value={fileId}
              onChange={(e) => setFileId(e.target.value)}
              placeholder="Paste file_id from upload page"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-sm text-gray-400 mb-1 block">Target Column</label>
            <input
              value={targetColumn}
              onChange={(e) => setTargetColumn(e.target.value)}
              placeholder="e.g. species"
              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2 text-sm"
            />
          </div>
        </div>
        <button
          onClick={handleTrain}
          disabled={!fileId || !targetColumn || loading}
          className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed py-3 rounded-lg font-medium transition-colors"
        >
          {loading ? "Training... (this takes 1-2 minutes)" : "Start Training"}
        </button>
      </div>

      {/* Results */}
      {result && (
        <>
          {/* Best Model Card */}
          <div className="bg-gradient-to-r from-blue-900/50 to-purple-900/50 rounded-xl p-6 border border-blue-800">
            <div className="flex items-center gap-3 mb-4">
              <Trophy className="text-yellow-400" size={28} />
              <h2 className="text-xl font-bold">Best Model: {result.best_model_name.replace(/_/g, " ")}</h2>
            </div>
            <div className="grid grid-cols-5 gap-3">
              {Object.entries(result.best_metrics).map(([key, val]) => (
                <div key={key} className="bg-gray-900/50 rounded-lg p-3 text-center">
                  <p className="text-gray-400 text-xs mb-1">{key.toUpperCase()}</p>
                  <p className="text-lg font-bold text-green-400">
                    {typeof val === "number" ? (val * 100).toFixed(1) + "%" : val}
                  </p>
                </div>
              ))}
            </div>
          </div>

          {/* Model Comparison Chart */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-lg font-semibold mb-6">Model Comparison</h2>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData}>
                <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <YAxis domain={[0, 100]} tick={{ fill: "#9ca3af", fontSize: 12 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
                  formatter={(val: any) => `${val}%`}
                />
                <Legend />
                <Bar dataKey="F1" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Accuracy" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                <Bar dataKey="ROC-AUC" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed Table */}
          <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
            <h2 className="text-lg font-semibold mb-4">Detailed Results</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-gray-400 border-b border-gray-800">
                    <th className="text-left py-2 pr-4">Model</th>
                    <th className="text-right py-2 pr-4">F1</th>
                    <th className="text-right py-2 pr-4">Accuracy</th>
                    <th className="text-right py-2 pr-4">Precision</th>
                    <th className="text-right py-2 pr-4">Recall</th>
                    <th className="text-right py-2 pr-4">ROC-AUC</th>
                    <th className="text-right py-2">Train Time</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.all_results).map(([name, data]: [string, any]) => (
                    <tr
                      key={name}
                      className={`border-b border-gray-800 ${name === result.best_model_name ? "text-green-400" : ""}`}
                    >
                      <td className="py-3 pr-4 font-medium flex items-center gap-2">
                        {name === result.best_model_name && <Trophy size={14} className="text-yellow-400" />}
                        {name.replace(/_/g, " ")}
                      </td>
                      <td className="text-right pr-4">{(data.metrics.f1 * 100).toFixed(1)}%</td>
                      <td className="text-right pr-4">{(data.metrics.accuracy * 100).toFixed(1)}%</td>
                      <td className="text-right pr-4">{(data.metrics.precision * 100).toFixed(1)}%</td>
                      <td className="text-right pr-4">{(data.metrics.recall * 100).toFixed(1)}%</td>
                      <td className="text-right pr-4">{(data.metrics.roc_auc * 100).toFixed(1)}%</td>
                      <td className="text-right text-gray-400">
                        <span className="flex items-center justify-end gap-1">
                          <Clock size={12} />
                          {data.metrics.training_time_seconds}s
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
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