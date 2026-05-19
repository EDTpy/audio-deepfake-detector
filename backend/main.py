# flake8: noqa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import io
import logging
from scipy import stats
from scipy.signal import find_peaks
import os

app = FastAPI(title="Audio Deepfake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174", 
        "https://audio-deepfake-detector-xi.vercel.app",
        "https://*.vercel.app",
        "*"
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  ADVANCED FEATURE EXTRACTION (Full Version with Safe Defaults)
# ─────────────────────────────────────────────────────────────

def extract_advanced_features(audio: np.ndarray, sr: int) -> dict:
    """Extract comprehensive features for AI voice detection with safe defaults"""
    
    # Initialize all features with default values
    default_features = {
        # MFCC features
        "mfcc_variance": 0.0,
        "mfcc_std_mean": 0.0,
        "mfcc_std_std": 0.0,
        "mfcc_delta_std": 0.0,
        "mfcc_delta2_std": 0.0,
        "mfcc_high_mean": 0.0,
        "mfcc_high_std": 0.0,
        
        # Pitch features
        "pitch_std": 0.0,
        "pitch_range": 0.0,
        "pitch_jitter": 0.0,
        "yin_std": 0.0,
        
        # Spectral features
        "centroid_mean": 0.0,
        "centroid_std": 0.0,
        "bandwidth_mean": 0.0,
        "bandwidth_std": 0.0,
        "rolloff_mean": 0.0,
        "rolloff_std": 0.0,
        "flatness_mean": 0.0,
        "flatness_std": 0.0,
        "contrast_mean": 0.0,
        "contrast_std": 0.0,
        
        # High-frequency features
        "hf_energy_mean": 0.0,
        "hf_energy_std": 0.0,
        "hf_variation": 0.0,
        "vhf_energy": 0.0,
        "spectral_slope": 0.0,
        
        # Temporal features
        "rms_std": 0.0,
        "rms_mean": 0.0,
        "zcr_std": 0.0,
        "zcr_mean": 0.0,
        "dynamic_range": 0.0,
        "energy_entropy": 0.0,
        
        # Artifact features
        "flux_std": 0.0,
        "flux_mean": 0.0,
        "spec_irregularity": 0.0,
        "avg_formant_width": 0.0,
        "formant_width_std": 0.0,
        "phase_consistency": 0.0,
        "low_mid_ratio": 0.0,
        "mid_high_ratio": 0.0,
    }
    
    try:
        # Ensure minimum length (3 seconds minimum for reliable analysis)
        min_samples = sr * 3
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        # Trim silence from beginning and end
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio_trimmed) > 0:
            audio = audio_trimmed
        
        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        
        hop_length = 256
        n_fft = 2048
        
        # ─────────────────────────────────────────────────────────
        # MFCC FEATURES (20 coefficients)
        # ─────────────────────────────────────────────────────────
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        
        # MFCC statistics - AI voices have less variation
        default_features["mfcc_variance"] = float(np.var(mfcc))
        default_features["mfcc_std_mean"] = float(np.std(mfcc, axis=1).mean())
        default_features["mfcc_std_std"] = float(np.std(mfcc, axis=1).std())
        default_features["mfcc_delta_std"] = float(np.std(mfcc_delta))
        default_features["mfcc_delta2_std"] = float(np.std(mfcc_delta2))
        default_features["mfcc_high_mean"] = float(mfcc[10:].mean()) if mfcc.shape[0] > 10 else 0.0
        default_features["mfcc_high_std"] = float(mfcc[10:].std()) if mfcc.shape[0] > 10 else 0.0
        
        # ─────────────────────────────────────────────────────────
        # PITCH FEATURES (Using multiple methods)
        # ─────────────────────────────────────────────────────────
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                if magnitudes[index, t] > 0.8:
                    pitch_values.append(pitches[index, t])
            
            if len(pitch_values) > 10:
                default_features["pitch_std"] = float(np.std(pitch_values))
                default_features["pitch_range"] = float(np.max(pitch_values) - np.min(pitch_values))
                
                # Pitch variation rate
                if len(pitch_values) > 20:
                    pitch_diff = np.diff(pitch_values)
                    default_features["pitch_jitter"] = float(np.std(pitch_diff) / (np.mean(np.abs(pitch_diff)) + 1e-8))
        except Exception as e:
            logger.warning(f"Pitch detection failed: {e}")
        
        # ─────────────────────────────────────────────────────────
        # SPECTRAL FEATURES
        # ─────────────────────────────────────────────────────────
        stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
        default_features["centroid_mean"] = float(np.mean(spectral_centroids))
        default_features["centroid_std"] = float(np.std(spectral_centroids))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
        default_features["bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        default_features["bandwidth_std"] = float(np.std(spectral_bandwidth))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
        default_features["rolloff_mean"] = float(np.mean(spectral_rolloff))
        default_features["rolloff_std"] = float(np.std(spectral_rolloff))
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
        default_features["flatness_mean"] = float(np.mean(spectral_flatness))
        default_features["flatness_std"] = float(np.std(spectral_flatness))
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
        default_features["contrast_mean"] = float(np.mean(spectral_contrast))
        default_features["contrast_std"] = float(np.std(spectral_contrast))
        
        # ─────────────────────────────────────────────────────────
        # HIGH-FREQUENCY ANALYSIS
        # ─────────────────────────────────────────────────────────
        hf_mask = freqs > 6000
        if hf_mask.any():
            default_features["hf_energy_mean"] = float(stft[hf_mask].mean())
            default_features["hf_energy_std"] = float(stft[hf_mask].std())
            default_features["hf_variation"] = float(np.std(stft[hf_mask], axis=1).mean()) if stft[hf_mask].shape[0] > 1 else 0.0
        
        vhf_mask = freqs > 10000
        if vhf_mask.any():
            default_features["vhf_energy"] = float(stft[vhf_mask].mean())
        
        # ─────────────────────────────────────────────────────────
        # TEMPORAL FEATURES
        # ─────────────────────────────────────────────────────────
        rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        default_features["rms_std"] = float(np.std(rms))
        default_features["rms_mean"] = float(np.mean(rms))
        rms_max = float(np.max(rms))
        rms_min = float(np.min(rms))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
        default_features["zcr_std"] = float(np.std(zcr))
        default_features["zcr_mean"] = float(np.mean(zcr))
        
        # Dynamic range
        if rms_min > 0:
            default_features["dynamic_range"] = float(20 * np.log10((rms_max + 1e-8) / (rms_min + 1e-8)))
        
        # Energy entropy
        rms_normalized = rms / (rms.sum() + 1e-8)
        default_features["energy_entropy"] = float(-np.sum(rms_normalized * np.log2(rms_normalized + 1e-8)))
        
        # ─────────────────────────────────────────────────────────
        # ARTIFACT DETECTION
        # ─────────────────────────────────────────────────────────
        # Spectral flux
        flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0))
        default_features["flux_std"] = float(np.std(flux))
        default_features["flux_mean"] = float(np.mean(flux))
        
        # Spectral irregularity
        mean_spec = np.mean(stft, axis=1) + 1e-8
        default_features["spec_irregularity"] = float(np.mean(np.abs(np.diff(mean_spec)) / mean_spec[:-1]))
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        # Return default features if extraction fails
    
    return default_features


# ─────────────────────────────────────────────────────────────
#  AGGRESSIVE AI CLASSIFIER
# ─────────────────────────────────────────────────────────────

def classify_audio(features: dict) -> tuple:
    """Aggressive classification specifically tuned for modern AI voices"""
    
    ai_score = 0
    human_score = 0
    details = []
    
    try:
        # 1. PITCH VARIATION
        pitch_std = features.get("pitch_std", 0)
        if pitch_std < 8 and pitch_std > 0:
            ai_score += 30
            details.append(f"⚠️ Very flat pitch ({pitch_std:.1f}Hz) - AI characteristic")
        elif pitch_std < 15 and pitch_std > 0:
            ai_score += 15
            details.append(f"⚠️ Limited pitch variation ({pitch_std:.1f}Hz) - suspicious")
        elif pitch_std > 25:
            human_score += 20
            details.append(f"✅ Good pitch variation ({pitch_std:.1f}Hz) - human-like")
        elif pitch_std > 18:
            human_score += 10
            details.append(f"✅ Natural pitch variation ({pitch_std:.1f}Hz)")
        elif pitch_std > 0:
            ai_score += 5
        
        # 2. MFCC CONSISTENCY
        mfcc_var = features.get("mfcc_variance", 0)
        if mfcc_var < 60 and mfcc_var > 0:
            ai_score += 25
            details.append(f"⚠️ Very consistent MFCCs ({mfcc_var:.1f}) - AI artifact")
        elif mfcc_var < 120 and mfcc_var > 0:
            ai_score += 12
            details.append(f"⚠️ Limited MFCC variation ({mfcc_var:.1f}) - suspicious")
        elif mfcc_var > 250:
            human_score += 15
            details.append(f"✅ Natural MFCC variation ({mfcc_var:.1f}) - human-like")
        elif mfcc_var > 180:
            human_score += 8
            details.append(f"✅ Good MFCC variation ({mfcc_var:.1f})")
        elif mfcc_var > 0:
            ai_score += 3
        
        # 3. DYNAMIC RANGE
        dyn_range = features.get("dynamic_range", 0)
        if dyn_range < 12 and dyn_range > 0:
            ai_score += 20
            details.append(f"⚠️ Highly compressed dynamics ({dyn_range:.1f}dB) - AI processing")
        elif dyn_range < 20 and dyn_range > 0:
            ai_score += 10
            details.append(f"⚠️ Limited dynamic range ({dyn_range:.1f}dB) - suspicious")
        elif dyn_range > 35:
            human_score += 15
            details.append(f"✅ Excellent dynamic range ({dyn_range:.1f}dB) - human-like")
        elif dyn_range > 28:
            human_score += 8
            details.append(f"✅ Good dynamic range ({dyn_range:.1f}dB)")
        elif dyn_range > 0:
            ai_score += 3
        
        # 4. SPECTRAL FLATNESS
        flatness = features.get("flatness_mean", 0)
        if flatness < 0.008 and flatness > 0:
            ai_score += 18
            details.append(f"⚠️ Too clean spectral flatness ({flatness:.5f}) - AI synthesis")
        elif flatness < 0.015 and flatness > 0:
            ai_score += 8
            details.append(f"⚠️ Unnaturally clean ({flatness:.5f}) - possible AI")
        elif flatness > 0.15:
            ai_score += 8
            details.append(f"⚠️ Unnatural spectral flatness ({flatness:.5f}) - artifact")
        elif 0.025 < flatness < 0.08:
            human_score += 12
            details.append(f"✅ Natural spectral flatness ({flatness:.5f}) - human-like")
        elif 0.018 < flatness < 0.12:
            human_score += 5
        elif flatness > 0:
            ai_score += 2
        
        # 5. ENERGY VARIATION
        rms_std = features.get("rms_std", 0)
        if rms_std < 0.02 and rms_std > 0:
            ai_score += 18
            details.append(f"⚠️ Very stable energy ({rms_std:.4f}) - AI characteristic")
        elif rms_std < 0.035 and rms_std > 0:
            ai_score += 8
            details.append(f"⚠️ Limited energy variation ({rms_std:.4f}) - suspicious")
        elif rms_std > 0.07:
            human_score += 12
            details.append(f"✅ Natural energy variation ({rms_std:.4f}) - human-like")
        elif rms_std > 0.05:
            human_score += 6
            details.append(f"✅ Good energy variation ({rms_std:.4f})")
        elif rms_std > 0:
            ai_score += 2
        
        # 6. HIGH-FREQUENCY CONTENT
        hf_energy = features.get("hf_energy_mean", 0)
        hf_var = features.get("hf_variation", 0)
        if hf_energy < 0.005 and hf_energy > 0:
            ai_score += 15
            details.append("⚠️ Missing high frequencies - bandlimited AI")
        elif hf_energy < 0.01 and hf_energy > 0:
            ai_score += 7
            details.append("⚠️ Very limited high frequencies - possible AI")
        elif hf_var < 0.0015 and hf_var > 0:
            ai_score += 10
            details.append("⚠️ Unnatural high-frequency consistency - AI artifact")
        elif hf_var < 0.003 and hf_var > 0:
            ai_score += 4
        elif hf_energy > 0.02 and hf_var > 0.005:
            human_score += 10
            details.append("✅ Natural high-frequency content - human-like")
        else:
            human_score += 3
        
        # 7. SPECTRAL FLUX
        flux_std = features.get("flux_std", 0)
        if flux_std < 12 and flux_std > 0:
            ai_score += 15
            details.append(f"⚠️ Very smooth spectral transitions ({flux_std:.1f}) - AI")
        elif flux_std < 25 and flux_std > 0:
            ai_score += 7
            details.append(f"⚠️ Limited spectral variation ({flux_std:.1f}) - suspicious")
        elif flux_std > 60:
            human_score += 10
            details.append(f"✅ Natural spectral variation ({flux_std:.1f}) - human-like")
        elif flux_std > 45:
            human_score += 5
        elif flux_std > 0:
            ai_score += 2
        
        # 8. ENERGY ENTROPY
        entropy = features.get("energy_entropy", 0)
        if entropy < 4.0 and entropy > 0:
            ai_score += 12
            details.append(f"⚠️ Very predictable energy ({entropy:.2f}) - AI pattern")
        elif entropy < 5.5 and entropy > 0:
            ai_score += 5
            details.append(f"⚠️ Somewhat predictable energy ({entropy:.2f}) - suspicious")
        elif entropy > 7.0:
            human_score += 8
            details.append(f"✅ Natural energy unpredictability ({entropy:.2f}) - human-like")
        elif entropy > 6.0:
            human_score += 4
        
        # 9. MFCC DELTA VARIATION
        delta_var = features.get("mfcc_delta_std", 0)
        if delta_var < 3.5 and delta_var > 0:
            ai_score += 12
            details.append(f"⚠️ Very smooth MFCC transitions ({delta_var:.1f}) - AI")
        elif delta_var < 5.5 and delta_var > 0:
            ai_score += 5
        elif delta_var > 9:
            human_score += 8
            details.append(f"✅ Natural MFCC dynamics ({delta_var:.1f})")
        
        # 10. ZCR VARIATION
        zcr_std = features.get("zcr_std", 0)
        if zcr_std < 0.018 and zcr_std > 0:
            ai_score += 10
            details.append("⚠️ Limited consonant variation - AI speech pattern")
        elif zcr_std < 0.03 and zcr_std > 0:
            ai_score += 4
        elif zcr_std > 0.07:
            human_score += 8
            details.append("✅ Rich consonant variation - human-like")
        elif zcr_std > 0.05:
            human_score += 4
        
        # 11. SPECTRAL IRREGULARITY
        irregularity = features.get("spec_irregularity", 0)
        if irregularity < 0.12 and irregularity > 0:
            ai_score += 10
            details.append(f"⚠️ Too smooth spectral pattern ({irregularity:.3f}) - AI")
        elif irregularity < 0.25 and irregularity > 0:
            ai_score += 4
        elif irregularity > 0.6:
            human_score += 6
        elif irregularity > 0.4:
            human_score += 3
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        # Return default verdict if classification fails
        return "FAKE", 0.30, 0.70, ["Analysis encountered an error"]
    
    # Calculate final percentages
    total_score = ai_score + human_score
    if total_score > 0:
        ai_percentage = (ai_score / total_score) * 100
        human_percentage = (human_score / total_score) * 100
    else:
        ai_percentage = 50
        human_percentage = 50
    
    # Determine verdict and confidences
    if ai_percentage > human_percentage:
        verdict = "FAKE"
        confidence = min(0.95, 0.5 + (ai_percentage - 50) / 100)
        prob_fake = round(confidence, 4)
        prob_real = round(1 - prob_fake, 4)
    else:
        verdict = "REAL"
        confidence = min(0.95, 0.5 + (human_percentage - 50) / 100)
        prob_real = round(confidence, 4)
        prob_fake = round(1 - prob_real, 4)
    
    # Ensure decisive verdict
    if verdict == "REAL" and prob_real < 0.51:
        prob_real = 0.51
        prob_fake = 0.49
    elif verdict == "FAKE" and prob_fake < 0.51:
        prob_fake = 0.51
        prob_real = 0.49
    
    # Log detailed analysis
    logger.info(f"=== AI Detection Results ===")
    logger.info(f"AI Score: {ai_score}, Human Score: {human_score}")
    logger.info(f"AI: {ai_percentage:.1f}% | Human: {human_percentage:.1f}%")
    logger.info(f"Verdict: {verdict} (Real: {prob_real:.1%}, AI: {prob_fake:.1%})")
    
    return verdict, prob_real, prob_fake, details[:6]


# ─────────────────────────────────────────────────────────────
#  ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"message": "Audio Deepfake Detector API - Advanced AI Voice Detection"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    logger.info(f"Analyzing: {file.filename}")
    
    contents = await file.read()
    size_kb = round(len(contents) / 1024, 2)
    
    if len(contents) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 25MB.")
    
    try:
        audio, sr = librosa.load(
            io.BytesIO(contents),
            sr=22050,
            mono=True,
            duration=30
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {str(e)}")
    
    duration = round(len(audio) / sr, 2)
    
    if duration < 2.0:
        raise HTTPException(status_code=422, detail=f"Audio too short ({duration}s). Need at least 2 seconds.")
    
    try:
        features = extract_advanced_features(audio, sr)
        verdict, conf_real, conf_fake, indicators = classify_audio(features)
        
        return {
            "filename": file.filename,
            "size_kb": size_kb,
            "duration_sec": duration,
            "verdict": verdict,
            "confidence_real": conf_real,
            "confidence_fake": conf_fake,
            "method": "ai_optimized_v4",
            "features": {
                "pitch_variation": f"{features.get('pitch_std', 0):.1f}",
                "mfcc_variance": round(features.get('mfcc_variance', 0), 2),
                "energy_variation": round(features.get('rms_std', 0), 4),
                "dynamic_range": f"{features.get('dynamic_range', 0):.1f}",
                "spectral_flatness": round(features.get('flatness_mean', 0), 5),
            },
            "indicators": indicators
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
        