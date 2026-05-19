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
import traceback

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
#  SAFE FEATURE EXTRACTION (with fallbacks for every key)
# ─────────────────────────────────────────────────────────────

def safe_get(dictionary, key, default=0.0):
    """Safely get a value from dictionary, return default if key missing or value is None"""
    try:
        val = dictionary.get(key, default)
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return default
        return val
    except:
        return default

def extract_advanced_features(audio: np.ndarray, sr: int) -> dict:
    """Extract comprehensive features for AI voice detection with safe defaults for EVERY key"""
    
    # Initialize EVERY feature with default value - this prevents missing key errors
    features = {
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
        # Ensure minimum length
        min_samples = sr * 3
        if len(audio) < min_samples:
            audio = np.pad(audio, (0, min_samples - len(audio)))
        
        # Trim silence
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
        if len(audio_trimmed) > 0:
            audio = audio_trimmed
        
        # Normalize
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak
        
        hop_length = 256
        n_fft = 2048
        
        # MFCC FEATURES
        try:
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            
            features["mfcc_variance"] = float(np.var(mfcc))
            features["mfcc_std_mean"] = float(np.std(mfcc, axis=1).mean())
            features["mfcc_std_std"] = float(np.std(mfcc, axis=1).std())
            features["mfcc_delta_std"] = float(np.std(mfcc_delta))
            features["mfcc_delta2_std"] = float(np.std(mfcc_delta2))
            
            if mfcc.shape[0] > 10:
                features["mfcc_high_mean"] = float(mfcc[10:].mean())
                features["mfcc_high_std"] = float(mfcc[10:].std())
        except Exception as e:
            logger.warning(f"MFCC extraction failed: {e}")
        
        # PITCH FEATURES
        try:
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                if magnitudes[index, t] > 0.8:
                    pitch_values.append(pitches[index, t])
            
            if len(pitch_values) > 10:
                features["pitch_std"] = float(np.std(pitch_values))
                features["pitch_range"] = float(np.max(pitch_values) - np.min(pitch_values))
                
                if len(pitch_values) > 20:
                    pitch_diff = np.diff(pitch_values)
                    features["pitch_jitter"] = float(np.std(pitch_diff) / (np.mean(np.abs(pitch_diff)) + 1e-8))
        except Exception as e:
            logger.warning(f"Pitch extraction failed: {e}")
        
        # SPECTRAL FEATURES
        try:
            stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
            features["centroid_mean"] = float(np.mean(spectral_centroids))
            features["centroid_std"] = float(np.std(spectral_centroids))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
            features["bandwidth_mean"] = float(np.mean(spectral_bandwidth))
            features["bandwidth_std"] = float(np.std(spectral_bandwidth))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
            features["rolloff_mean"] = float(np.mean(spectral_rolloff))
            features["rolloff_std"] = float(np.std(spectral_rolloff))
            
            spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
            features["flatness_mean"] = float(np.mean(spectral_flatness))
            features["flatness_std"] = float(np.std(spectral_flatness))
            
            spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
            features["contrast_mean"] = float(np.mean(spectral_contrast))
            features["contrast_std"] = float(np.std(spectral_contrast))
            
            # High-frequency analysis
            hf_mask = freqs > 6000
            if hf_mask.any():
                features["hf_energy_mean"] = float(stft[hf_mask].mean())
                features["hf_energy_std"] = float(stft[hf_mask].std())
                if stft[hf_mask].shape[0] > 1:
                    features["hf_variation"] = float(np.std(stft[hf_mask], axis=1).mean())
            
            vhf_mask = freqs > 10000
            if vhf_mask.any():
                features["vhf_energy"] = float(stft[vhf_mask].mean())
            
            # Spectral flux
            flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0))
            features["flux_std"] = float(np.std(flux))
            features["flux_mean"] = float(np.mean(flux))
            
            # Spectral irregularity
            mean_spec = np.mean(stft, axis=1) + 1e-8
            features["spec_irregularity"] = float(np.mean(np.abs(np.diff(mean_spec)) / mean_spec[:-1]))
            
        except Exception as e:
            logger.warning(f"Spectral extraction failed: {e}")
        
        # TEMPORAL FEATURES
        try:
            rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
            features["rms_std"] = float(np.std(rms))
            features["rms_mean"] = float(np.mean(rms))
            rms_max = float(np.max(rms))
            rms_min = float(np.min(rms))
            
            zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
            features["zcr_std"] = float(np.std(zcr))
            features["zcr_mean"] = float(np.mean(zcr))
            
            if rms_min > 0:
                features["dynamic_range"] = float(20 * np.log10((rms_max + 1e-8) / (rms_min + 1e-8)))
            
            rms_normalized = rms / (rms.sum() + 1e-8)
            features["energy_entropy"] = float(-np.sum(rms_normalized * np.log2(rms_normalized + 1e-8)))
            
        except Exception as e:
            logger.warning(f"Temporal extraction failed: {e}")
            
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        logger.error(traceback.format_exc())
        # Return default features (all zeros)
    
    return features


# ─────────────────────────────────────────────────────────────
#  SAFE CLASSIFIER (uses safe_get for every access)
# ─────────────────────────────────────────────────────────────

def classify_audio(features: dict) -> tuple:
    """Safe classification with fallbacks for every feature access"""
    
    ai_score = 0
    human_score = 0
    details = []
    
    try:
        # 1. PITCH VARIATION - using safe_get
        pitch_std = safe_get(features, "pitch_std", 0)
        if pitch_std > 0:
            if pitch_std < 8:
                ai_score += 30
                details.append(f"⚠️ Very flat pitch ({pitch_std:.1f}Hz) - AI characteristic")
            elif pitch_std < 15:
                ai_score += 15
                details.append(f"⚠️ Limited pitch variation ({pitch_std:.1f}Hz) - suspicious")
            elif pitch_std > 25:
                human_score += 20
                details.append(f"✅ Good pitch variation ({pitch_std:.1f}Hz) - human-like")
            elif pitch_std > 18:
                human_score += 10
                details.append(f"✅ Natural pitch variation ({pitch_std:.1f}Hz)")
            else:
                ai_score += 5
        else:
            details.append("ℹ️ Pitch detection limited - analysis continues")
        
        # 2. MFCC CONSISTENCY
        mfcc_var = safe_get(features, "mfcc_variance", 0)
        if mfcc_var > 0:
            if mfcc_var < 60:
                ai_score += 25
                details.append(f"⚠️ Very consistent MFCCs ({mfcc_var:.1f}) - AI artifact")
            elif mfcc_var < 120:
                ai_score += 12
                details.append(f"⚠️ Limited MFCC variation ({mfcc_var:.1f}) - suspicious")
            elif mfcc_var > 250:
                human_score += 15
                details.append(f"✅ Natural MFCC variation ({mfcc_var:.1f}) - human-like")
            elif mfcc_var > 180:
                human_score += 8
                details.append(f"✅ Good MFCC variation ({mfcc_var:.1f})")
            else:
                ai_score += 3
        
        # 3. DYNAMIC RANGE
        dyn_range = safe_get(features, "dynamic_range", 0)
        if dyn_range > 0:
            if dyn_range < 12:
                ai_score += 20
                details.append(f"⚠️ Highly compressed dynamics ({dyn_range:.1f}dB) - AI processing")
            elif dyn_range < 20:
                ai_score += 10
                details.append(f"⚠️ Limited dynamic range ({dyn_range:.1f}dB) - suspicious")
            elif dyn_range > 35:
                human_score += 15
                details.append(f"✅ Excellent dynamic range ({dyn_range:.1f}dB) - human-like")
            elif dyn_range > 28:
                human_score += 8
                details.append(f"✅ Good dynamic range ({dyn_range:.1f}dB)")
            else:
                ai_score += 3
        
        # 4. SPECTRAL FLATNESS
        flatness = safe_get(features, "flatness_mean", 0)
        if flatness > 0:
            if flatness < 0.008:
                ai_score += 18
                details.append(f"⚠️ Too clean spectral flatness ({flatness:.5f}) - AI synthesis")
            elif flatness < 0.015:
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
            else:
                ai_score += 2
        
        # 5. ENERGY VARIATION
        rms_std = safe_get(features, "rms_std", 0)
        if rms_std > 0:
            if rms_std < 0.02:
                ai_score += 18
                details.append(f"⚠️ Very stable energy ({rms_std:.4f}) - AI characteristic")
            elif rms_std < 0.035:
                ai_score += 8
                details.append(f"⚠️ Limited energy variation ({rms_std:.4f}) - suspicious")
            elif rms_std > 0.07:
                human_score += 12
                details.append(f"✅ Natural energy variation ({rms_std:.4f}) - human-like")
            elif rms_std > 0.05:
                human_score += 6
                details.append(f"✅ Good energy variation ({rms_std:.4f})")
            else:
                ai_score += 2
        
        # 6. HIGH-FREQUENCY CONTENT
        hf_energy = safe_get(features, "hf_energy_mean", 0)
        hf_var = safe_get(features, "hf_variation", 0)
        if hf_energy > 0:
            if hf_energy < 0.005:
                ai_score += 15
                details.append("⚠️ Missing high frequencies - bandlimited AI")
            elif hf_energy < 0.01:
                ai_score += 7
                details.append("⚠️ Very limited high frequencies - possible AI")
            elif hf_var < 0.0015:
                ai_score += 10
                details.append("⚠️ Unnatural high-frequency consistency - AI artifact")
            elif hf_var < 0.003:
                ai_score += 4
            elif hf_energy > 0.02 and hf_var > 0.005:
                human_score += 10
                details.append("✅ Natural high-frequency content - human-like")
            else:
                human_score += 3
        
        # 7. SPECTRAL FLUX
        flux_std = safe_get(features, "flux_std", 0)
        if flux_std > 0:
            if flux_std < 12:
                ai_score += 15
                details.append(f"⚠️ Very smooth spectral transitions ({flux_std:.1f}) - AI")
            elif flux_std < 25:
                ai_score += 7
                details.append(f"⚠️ Limited spectral variation ({flux_std:.1f}) - suspicious")
            elif flux_std > 60:
                human_score += 10
                details.append(f"✅ Natural spectral variation ({flux_std:.1f}) - human-like")
            elif flux_std > 45:
                human_score += 5
            else:
                ai_score += 2
        
        # 8. ENERGY ENTROPY
        entropy = safe_get(features, "energy_entropy", 0)
        if entropy > 0:
            if entropy < 4.0:
                ai_score += 12
                details.append(f"⚠️ Very predictable energy ({entropy:.2f}) - AI pattern")
            elif entropy < 5.5:
                ai_score += 5
                details.append(f"⚠️ Somewhat predictable energy ({entropy:.2f}) - suspicious")
            elif entropy > 7.0:
                human_score += 8
                details.append(f"✅ Natural energy unpredictability ({entropy:.2f}) - human-like")
            elif entropy > 6.0:
                human_score += 4
        
        # 9. MFCC DELTA VARIATION
        delta_var = safe_get(features, "mfcc_delta_std", 0)
        if delta_var > 0:
            if delta_var < 3.5:
                ai_score += 12
                details.append(f"⚠️ Very smooth MFCC transitions ({delta_var:.1f}) - AI")
            elif delta_var < 5.5:
                ai_score += 5
            elif delta_var > 9:
                human_score += 8
                details.append(f"✅ Natural MFCC dynamics ({delta_var:.1f})")
        
        # 10. ZCR VARIATION
        zcr_std = safe_get(features, "zcr_std", 0)
        if zcr_std > 0:
            if zcr_std < 0.018:
                ai_score += 10
                details.append("⚠️ Limited consonant variation - AI speech pattern")
            elif zcr_std < 0.03:
                ai_score += 4
            elif zcr_std > 0.07:
                human_score += 8
                details.append("✅ Rich consonant variation - human-like")
            elif zcr_std > 0.05:
                human_score += 4
        
        # 11. SPECTRAL IRREGULARITY
        irregularity = safe_get(features, "spec_irregularity", 0)
        if irregularity > 0:
            if irregularity < 0.12:
                ai_score += 10
                details.append(f"⚠️ Too smooth spectral pattern ({irregularity:.3f}) - AI")
            elif irregularity < 0.25:
                ai_score += 4
            elif irregularity > 0.6:
                human_score += 6
            elif irregularity > 0.4:
                human_score += 3
        
        # 12. PITCH JITTER
        jitter = safe_get(features, "pitch_jitter", 0)
        if jitter > 0:
            if jitter < 0.3:
                ai_score += 8
                details.append(f"⚠️ Very low pitch jitter ({jitter:.2f}) - AI")
            elif jitter > 0.8:
                human_score += 6
                
    except Exception as e:
        logger.error(f"Classification error: {e}")
        logger.error(traceback.format_exc())
        # Return a safe default
        return "FAKE", 0.30, 0.70, ["Analysis encountered an error - please try again"]
    
    # Calculate percentages (safe division)
    total_score = ai_score + human_score
    if total_score > 0:
        ai_percentage = (ai_score / total_score) * 100
        human_percentage = (human_score / total_score) * 100
    else:
        ai_percentage = 50
        human_percentage = 50
    
    # Determine verdict based on higher percentage
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
    
    # Ensure valid probabilities
    prob_real = max(0.01, min(0.99, prob_real))
    prob_fake = max(0.01, min(0.99, prob_fake))
    
    # Log results
    logger.info(f"=== Detection Results ===")
    logger.info(f"AI Score: {ai_score}, Human Score: {human_score}")
    logger.info(f"AI: {ai_percentage:.1f}% | Human: {human_percentage:.1f}%")
    logger.info(f"Verdict: {verdict} (Real: {prob_real:.1%}, AI: {prob_fake:.1%})")
    
    return verdict, prob_real, prob_fake, details[:5]


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
    
    try:
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
        
        # Extract features and classify with safety
        features = extract_advanced_features(audio, sr)
        verdict, conf_real, conf_fake, indicators = classify_audio(features)
        
        # Safe feature access for response
        pitch_val = safe_get(features, "pitch_std", 0)
        mfcc_val = safe_get(features, "mfcc_variance", 0)
        rms_val = safe_get(features, "rms_std", 0)
        dyn_val = safe_get(features, "dynamic_range", 0)
        flat_val = safe_get(features, "flatness_mean", 0)
        
        return {
            "filename": file.filename,
            "size_kb": size_kb,
            "duration_sec": duration,
            "verdict": verdict,
            "confidence_real": conf_real,
            "confidence_fake": conf_fake,
            "method": "safe_v1",
            "features": {
                "pitch_variation": f"{pitch_val:.1f}",
                "mfcc_variance": round(mfcc_val, 2),
                "energy_variation": round(rms_val, 4),
                "dynamic_range": f"{dyn_val:.1f}",
                "spectral_flatness": round(flat_val, 5),
            },
            "indicators": indicators
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    
    
        