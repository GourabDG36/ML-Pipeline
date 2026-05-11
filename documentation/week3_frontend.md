# Week 3 - Next.js Frontend

## What We Built
- 4-page Next.js frontend with Tailwind CSS
- Upload page with drag-and-drop CSV upload
- Train page with model comparison chart and metrics table
- Predict page with SHAP feature importance chart
- Drift Monitor page with per-feature drift visualization
- Axios API client with TypeScript types
- localStorage for passing file_id between pages

---

## Tech Stack
- Next.js 14 with App Router
- TypeScript
- Tailwind CSS
- Recharts for charts
- Lucide React for icons
- Axios for API calls

---

## Files Created

### `frontend/lib/api.ts`
- Axios client with 10 minute timeout for training
- TypeScript interfaces for all API responses
- Functions: uploadCSV, validateCSV, trainModels, predict, getDriftReport
- Base URL from NEXT_PUBLIC_API_URL env variable

### `frontend/app/layout.tsx`
- Global layout with navigation bar
- Links to all 4 pages

### `frontend/app/page.tsx` (Upload page)
- File picker with click to upload
- Shows rows, columns, file_id after upload
- Target column dropdown
- Validation results with errors/warnings
- Feature type badges (numerical/categorical)
- Saves file_id, target_column, features to localStorage

### `frontend/app/train/page.tsx` (Train page)
- Auto-loads file_id and target_column from localStorage
- Start Training button with loading state
- Best model card with all metrics
- Bar chart comparing all 4 models (F1, Accuracy, ROC-AUC)
- Detailed metrics table with training time

### `frontend/app/predict/page.tsx` (Predict page)
- Auto-loads file_id and features from localStorage
- Dynamic input fields for each feature
- Predicted class with confidence percentage
- Probability bars for each class
- Horizontal SHAP bar chart

### `frontend/app/drift/page.tsx` (Drift Monitor page)
- File ID input (auto-loaded from localStorage)
- JSON textarea for current data input
- Status card (green/red based on drift detected)
- Drift score, features drifted, current rows stats
- Per-feature drift bars with stat test info

### `frontend/.env.local`
- NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1

---

## How to Run

### Development
```powershell
# Terminal 1 - Backend
cd D:\my projects\mini_project\ml-pipeline
docker-compose up

# Terminal 2 - Frontend
cd D:\my projects\mini_project\ml-pipeline\frontend
npm run dev
```

### URLs
- http://localhost:3000 - Frontend
- http://localhost:8000/docs - Backend API
- http://localhost:5000 - MLflow UI

---

## Key Problems Solved

### Problem 1: CORS blocking requests from frontend
- Cause: Docker backend not sending CORS headers
- Fix: Updated allow_origins in FastAPI CORS middleware to include localhost:3000

### Problem 2: Training timeout
- Cause: Optuna 20 trials too slow inside Docker
- Fix: Reduced OPTUNA_TRIALS to 5 in docker-compose.yml environment
- Fix: Increased axios timeout to 600000ms (10 minutes)

### Problem 3: MLflow version mismatch
- Cause: FastAPI container had newer MLflow calling endpoints that MLflow 2.13.0 server didn't have
- Fix: Pinned mlflow==2.13.0 in requirements.docker.txt to match server version

### Problem 4: Docker layer cache corruption
- Cause: Interrupted builds corrupt Docker's layer cache
- Fix: docker system prune -f then docker-compose up --build

### Problem 5: MLflow not ready when FastAPI starts
- Cause: FastAPI starts before MLflow finishes installing and starting
- Fix: Added retry loop in lifespan that waits up to 5 minutes for MLflow

---

## User Flow

```
1. Upload page
   Upload CSV -> Get file_id
   Select target column -> Validate
   file_id saved to localStorage
        |
        v
2. Train page
   file_id auto-loaded
   Click Train -> Wait 1-2 minutes
   See model comparison chart
        |
        v
3. Predict page
   file_id + features auto-loaded
   Enter feature values
   See prediction + SHAP chart
        |
        v
4. Drift Monitor page
   file_id auto-loaded
   Paste JSON array of recent data
   See per-feature drift scores
```

---

## Test Results with Iris Dataset

### Upload
- 150 rows, 5 columns detected
- species selected as target
- 4 numerical features detected
- 1 duplicate warning

### Training
- Best model: Logistic Regression
- F1: 96.7%, Accuracy: 96.7%, ROC-AUC: 100%
- All 4 models compared in chart

### Prediction (setosa sample)
- Predicted: setosa
- Confidence: 99.3%
- SHAP shows petal dimensions most important

### Drift Monitor (10 rows with extreme 99.0 values)
- Status: No Drift Detected
- Drift Score: 0.5
- All 4 features shown with K-S scores
