================================================================
  AUDIO DEEPFAKE DETECTOR — DEVELOPER SETUP GUIDE
  Last updated: April 2026
  For questions contact the project owner
================================================================

WHAT THIS PROJECT DOES
-----------------------
A web application that analyzes audio files and live recordings
to detect whether a voice is a real human or AI-generated
(ElevenLabs, OpenAI TTS, etc.).

Stack:
  - Backend  : Python + FastAPI
  - Frontend : React + Vite
  - ML Model : Transformer model from HuggingFace (~400MB download)
  - Audio    : librosa for feature extraction


================================================================
  STEP 1 — INSTALL REQUIRED SOFTWARE
================================================================

You need THREE things installed before anything else.
Check each one by opening PowerShell and running the command.

────────────────────────────────────────────────────────────────
1A. Python 3.11 or 3.12 (NOT 3.13 or 3.14 — too new)
────────────────────────────────────────────────────────────────
Check if installed:
  python --version

If not installed or version is 3.13/3.14:
  → Go to: https://www.python.org/downloads/release/python-3119/
  → Download "Windows installer (64-bit)"
  → During install: CHECK "Add Python to PATH" ✅
  → Click Install Now

IMPORTANT: If you have Python 3.14, you must install 3.11 or 3.12
alongside it and use "py -3.11" instead of "python" in all commands.

────────────────────────────────────────────────────────────────
1B. Node.js v18 or higher
────────────────────────────────────────────────────────────────
Check if installed:
  node --version

If not installed:
  → Go to: https://nodejs.org
  → Download the LTS version
  → Install with all default options

────────────────────────────────────────────────────────────────
1C. Git (optional but recommended)
────────────────────────────────────────────────────────────────
Check if installed:
  git --version

If not installed:
  → Go to: https://git-scm.com/download/win
  → Install with all default options


================================================================
  STEP 2 — EXTRACT THE PROJECT
================================================================

1. Extract the ZIP file you received to a location of your choice
   Example: C:\Users\YourName\Desktop\deepfake-detector

2. Open VS Code:
   File → Open Folder → select the "deepfake-detector" folder

3. Open the VS Code terminal:
   Terminal → New Terminal  (or press Ctrl + `)


================================================================
  STEP 3 — SET UP THE BACKEND
================================================================

In the VS Code terminal, run these commands ONE AT A TIME.
Wait for each to finish before running the next.

────────────────────────────────────────────────────────────────
Navigate to backend folder:
────────────────────────────────────────────────────────────────
  cd backend

────────────────────────────────────────────────────────────────
Install Python dependencies:
────────────────────────────────────────────────────────────────
  python -m pip install --only-binary :all: fastapi uvicorn python-multipart python-dotenv numpy librosa soundfile scipy

  python -m pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cpu

  python -m pip install --only-binary :all: transformers huggingface_hub

NOTE: torch is about 200MB. transformers is about 50MB.
      This step requires internet and may take 5-10 minutes.

────────────────────────────────────────────────────────────────
Start the backend server:
────────────────────────────────────────────────────────────────
  python -m uvicorn main:app --reload --port 8000

You should see:
  INFO: Application startup complete.
  INFO: Uvicorn running on http://127.0.0.1:8000

AND one of these two lines:
  ✅ ML model loaded — running at full accuracy
  ⚠  ML model unavailable. Using heuristic fallback.

The first time you run this, the ML model (~400MB) downloads
automatically from HuggingFace. This takes 2-5 minutes.
After the first run it is cached and loads instantly.

KEEP THIS TERMINAL OPEN. The backend must stay running.


================================================================
  STEP 4 — SET UP THE FRONTEND
================================================================

Open a SECOND terminal in VS Code:
  Terminal → New Terminal  (or click the + button)

Run these commands:

  cd frontend
  npm install
  npm run dev

You should see:
  VITE v6.x.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/

Open your browser and go to: http://localhost:5173


================================================================
  STEP 5 — TEST THE APP
================================================================

The app has two modes:

📁 UPLOAD FILE
  - Click the "Upload File" tab
  - Drag and drop or click to browse for an audio file
  - Supported formats: WAV, MP3, OGG, FLAC, M4A
  - Maximum file size: 250MB
  - Click "Run Analysis"
  - Results appear on the right side

🎙️ RECORD LIVE
  - Click the "Record Live" tab
  - Click the ⏺ button to start recording
  - Allow microphone access when the browser asks
  - You will hear a start beep
  - Speak naturally for at least 2 seconds
  - Click ⏹ to stop — you will hear a stop beep
  - Your recording appears with a playback player
  - Click "Run Analysis" to detect

READING THE RESULTS:
  ✅ REAL VOICE   — The audio is a genuine human voice
  🚨 AI GENERATED — The audio shows signs of AI synthesis
  The confidence bars show the probability (0-100%)


================================================================
  COMMON ERRORS AND HOW TO FIX THEM
================================================================

ERROR: 'python' is not recognized
→ Python is not installed or not on PATH
→ Reinstall Python and check "Add to PATH" during setup
→ Try "py" instead of "python"

ERROR: 'npm' is not recognized
→ Node.js is not installed
→ Download and install from https://nodejs.org

ERROR: 'vite' is not recognized
→ Run "npm install" in the frontend folder first

ERROR: ModuleNotFoundError: No module named 'xxx'
→ Run: python -m pip install --only-binary :all: xxx
→ Replace "xxx" with the module name from the error

ERROR: Cannot reach the backend server
→ Make sure the backend terminal is still running
→ Check that it says "Uvicorn running on http://127.0.0.1:8000"
→ If it crashed, run: python -m uvicorn main:app --reload --port 8000

ERROR: 422 Unprocessable Content (on recording)
→ The audio format could not be decoded
→ Try uploading a WAV file instead of recording
→ Make sure you recorded for at least 2 seconds

ERROR: pip install fails with "no C compiler"
→ Always use --only-binary :all: flag
→ Example: python -m pip install --only-binary :all: numpy

ERROR: torch install fails
→ Run this exact command:
→ python -m pip install --only-binary :all: torch --index-url https://download.pytorch.org/whl/cpu

ERROR: Port 8000 already in use
→ Run: netstat -ano | findstr :8000
→ Then: taskkill /PID <number shown> /F
→ Then restart the backend

ERROR: Port 5173 already in use
→ Edit frontend/vite.config.js and change port: 5173 to port: 5174
→ Then run npm run dev again


================================================================
  PROJECT FILE STRUCTURE
================================================================

deepfake-detector/
│
├── backend/
│   ├── main.py              ← FastAPI server + ML model + analysis
│   └── requirements.txt     ← Python package list
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx           ← Main app layout
│   │   ├── index.css         ← All styles
│   │   └── components/
│   │       ├── Header.jsx    ← Top navigation bar
│   │       ├── Uploader.jsx  ← File upload + waveform viewer
│   │       ├── Recorder.jsx  ← Live microphone recorder
│   │       └── Verdict.jsx   ← Results card with confidence scores
│   ├── index.html
│   ├── package.json          ← Node dependencies
│   └── vite.config.js        ← Dev server config (proxy to port 8000)
│
└── README.txt               ← This file


================================================================
  PYTHON PACKAGES (what they do)
================================================================

fastapi          — Web API framework
uvicorn          — Web server that runs FastAPI
python-multipart — Handles file uploads
python-dotenv    — Loads environment variables
numpy            — Numerical computing
librosa          — Audio analysis (MFCCs, spectral features)
soundfile        — Audio file reading/writing
scipy            — Scientific computing (used by librosa)
torch            — PyTorch deep learning framework
transformers     — HuggingFace model loader
huggingface_hub  — Downloads models from HuggingFace


================================================================
  NODE PACKAGES (what they do)
================================================================

react            — UI framework
react-dom        — React browser renderer
react-router-dom — Page routing (Analyze / About pages)
axios            — HTTP requests to the backend
wavesurfer.js    — Audio waveform visualization
lucide-react     — Icons


================================================================
  HOW THE DETECTION WORKS
================================================================

1. Audio is uploaded or recorded in the browser
2. Frontend sends the audio file to the backend at POST /analyze
3. Backend loads the audio with librosa at 16kHz mono
4. If the ML model is loaded:
     - Audio is fed to a HuggingFace transformer model
       (mo-thecreator/deepfake-audio-detection)
     - Model was fine-tuned on real vs. AI-generated speech
     - Returns probability scores for REAL and FAKE
5. If ML model unavailable (fallback):
     - Extracts 15+ acoustic features (MFCCs, pitch, spectral flux)
     - Scores each feature against research-backed thresholds
     - Weighted voting system produces final confidence score
6. Result returned to frontend with verdict + confidence bars


================================================================
  QUICK START CHECKLIST
================================================================

[ ] Python 3.11 or 3.12 installed
[ ] Node.js 18+ installed
[ ] Project folder extracted
[ ] Terminal 1: cd backend → pip installs → python -m uvicorn main:app --reload --port 8000
[ ] Terminal 2: cd frontend → npm install → npm run dev
[ ] Browser open at http://localhost:5173
[ ] Backend shows "ML model loaded" or "heuristic fallback"
[ ] Upload a test audio file and click Run Analysis


================================================================
  NOTES FOR THE DEVELOPER
================================================================

- The ML model is cached after first download at:
  C:\Users\<username>\.cache\huggingface\hub\
  Delete this folder to force a fresh download.

- The backend must be running for the frontend to work.
  If you close the backend terminal, the frontend will show
  "Cannot reach backend server".

- The frontend proxies /analyze and /health to port 8000
  via vite.config.js. Do not change the backend port without
  also updating vite.config.js.

- Audio recorded in the browser is converted to WAV format
  before being sent to the backend. This ensures librosa
  can always decode it correctly on Windows.

- For best detection accuracy, provide at least 3-5 seconds
  of clear speech with no background music.

================================================================
  END OF README
================================================================
