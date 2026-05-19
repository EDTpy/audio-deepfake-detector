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
#  ADVANCED FEATURE EXTRACTION (Full Version)
# ─────────────────────────────────────────────────────────────

def extract_advanced_features(audio: np.ndarray, sr: int) -> dict:
    """Extract comprehensive features for AI voice detection"""
    
    # Ensure minimum length (3 seconds minimum for reliable analysis)
    min_samples = sr * 3
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    # Trim silence from beginning and end
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio_trimmed) > 0:
        audio = audio_trimmed
    
    # Normalize
    audio = audio / (np.abs(audio).max() + 1e-8)
    
    hop_length = 256
    n_fft = 2048
    
    # ─────────────────────────────────────────────────────────
    # MFCC FEATURES (20 coefficients)
    # ─────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # MFCC statistics - AI voices have less variation
    mfcc_variance = float(np.var(mfcc))
    mfcc_std_mean = float(np.std(mfcc, axis=1).mean())
    mfcc_std_std = float(np.std(mfcc, axis=1).std())
    mfcc_delta_std = float(np.std(mfcc_delta))
    mfcc_delta2_std = float(np.std(mfcc_delta2))
    mfcc_high_mean = float(mfcc[10:].mean())
    mfcc_high_std = float(mfcc[10:].std())
    
    # ─────────────────────────────────────────────────────────
    # PITCH FEATURES (Using multiple methods)
    # ─────────────────────────────────────────────────────────
    try:
        # Method 1: piptrack
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=hop_length)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            if magnitudes[index, t] > 0.8:
                pitch_values.append(pitches[index, t])
        
        # Method 2: yin for comparison
        f0_yin = librosa.yin(audio, fmin=60, fmax=400, hop_length=hop_length)
        f0_yin = f0_yin[(f0_yin > 60) & (f0_yin < 400)]
        
        if len(pitch_values) > 10:
            pitch_std = float(np.std(pitch_values))
            pitch_range = float(np.max(pitch_values) - np.min(pitch_values))
            pitch_mean = float(np.mean(pitch_values))
            
            # Pitch variation rate (how quickly pitch changes)
            if len(pitch_values) > 20:
                pitch_diff = np.diff(pitch_values)
                pitch_jitter = float(np.std(pitch_diff) / (np.mean(np.abs(pitch_diff)) + 1e-8))
            else:
                pitch_jitter = 0
        else:
            pitch_std, pitch_range, pitch_mean, pitch_jitter = 0, 0, 0, 0
            
        if len(f0_yin) > 10:
            yin_std = float(np.std(f0_yin))
        else:
            yin_std = 0
            
    except Exception as e:
        logger.warning(f"Pitch detection failed: {e}")
        pitch_std, pitch_range, pitch_mean, pitch_jitter, yin_std = 0, 0, 0, 0, 0
    
    # ─────────────────────────────────────────────────────────
    # SPECTRAL FEATURES
    # ─────────────────────────────────────────────────────────
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    centroid_mean = float(np.mean(spectral_centroids))
    centroid_std = float(np.std(spectral_centroids))
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
    bandwidth_mean = float(np.mean(spectral_bandwidth))
    bandwidth_std = float(np.std(spectral_bandwidth))
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
    rolloff_mean = float(np.mean(spectral_rolloff))
    rolloff_std = float(np.std(spectral_rolloff))
    
    # Spectral flatness (tonal vs noise)
    spectral_flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
    flatness_mean = float(np.mean(spectral_flatness))
    flatness_std = float(np.std(spectral_flatness))
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, hop_length=hop_length)
    contrast_mean = float(np.mean(spectral_contrast))
    contrast_std = float(np.std(spectral_contrast))
    
    # ─────────────────────────────────────────────────────────
    # HIGH-FREQUENCY ANALYSIS
    # ─────────────────────────────────────────────────────────
    hf_mask = freqs > 6000
    vhf_mask = freqs > 10000
    
    if hf_mask.any():
        hf_energy_mean = float(stft[hf_mask].mean())
        hf_energy_std = float(stft[hf_mask].std())
        hf_variation = float(np.std(stft[hf_mask], axis=1).mean())
    else:
        hf_energy_mean, hf_energy_std, hf_variation = 0, 0, 0
    
    if vhf_mask.any():
        vhf_energy = float(stft[vhf_mask].mean())
    else:
        vhf_energy = 0
    
    # Spectral slope (AI often has unnatural rolloff)
    if len(freqs) > 10 and stft.shape[0] > 10:
        mean_spectrum = np.mean(stft, axis=1)
        log_freqs = np.log10(freqs[1:11] + 1)
        log_energy = np.log10(mean_spectrum[1:11] + 1e-8)
        if len(log_freqs) == len(log_energy):
            slope, _, _, _, _ = stats.linregress(log_freqs, log_energy)
            spectral_slope = float(slope)
        else:
            spectral_slope = 0
    else:
        spectral_slope = 0
    
    # ─────────────────────────────────────────────────────────
    # TEMPORAL FEATURES
    # ─────────────────────────────────────────────────────────
    # RMS energy
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    rms_std = float(np.std(rms))
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))
    rms_min = float(np.min(rms))
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)[0]
    zcr_std = float(np.std(zcr))
    zcr_mean = float(np.mean(zcr))
    
    # Dynamic range
    dynamic_range = float(20 * np.log10((rms_max + 1e-8) / (rms_min + 1e-8)))
    
    # Energy entropy (predictability of energy patterns)
    rms_normalized = rms / (rms.sum() + 1e-8)
    energy_entropy = float(-np.sum(rms_normalized * np.log2(rms_normalized + 1e-8)))
    
    # ─────────────────────────────────────────────────────────
    # ARTIFACT DETECTION (Specific to TTS/vocoders)
    # ─────────────────────────────────────────────────────────
    # Spectral flux (rate of spectral change)
    flux = np.sqrt(np.mean(np.diff(stft, axis=1) ** 2, axis=0))
    flux_std = float(np.std(flux))
    flux_mean = float(np.mean(flux))
    
    # Spectral irregularity (unnatural smoothness or roughness)
    mean_spec = np.mean(stft, axis=1) + 1e-8
    spec_irregularity = float(np.mean(np.abs(np.diff(mean_spec)) / mean_spec[:-1]))
    
    # Formant bandwidth (AI often has unnatural formants)
    try:
        formant_widths = []
        for frame in stft[:, :50].T:
            peaks, properties = find_peaks(frame, height=np.percentile(frame, 70), width=2)
            if len(peaks) > 0:
                widths = properties['widths']
                formant_widths.extend(widths)
        avg_formant_width = float(np.mean(formant_widths)) if formant_widths else 0
        formant_width_std = float(np.std(formant_widths)) if formant_widths else 0
    except:
        avg_formant_width, formant_width_std = 0, 0
    
    # Phase consistency (vocoders introduce phase artifacts)
    stft_complex = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(stft_complex)
    phase_consistency = float(np.std(phase))
    
    # Sub-band energy ratios
    bands = [(0, 500), (500, 2000), (2000, 6000), (6000, 12000), (12000, sr//2)]
    band_ratios = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            band_energy = stft[mask].mean()
            band_ratios.append(band_energy)
    
    if len(band_ratios) >= 2:
        low_mid_ratio = band_ratios[0] / (band_ratios[1] + 1e-8)
        mid_high_ratio = band_ratios[2] / (band_ratios[3] + 1e-8)
    else:
        low_mid_ratio, mid_high_ratio = 0, 0
    
    return {
        # MFCC features
        "mfcc_variance": mfcc_variance,
        "mfcc_std_mean": mfcc_std_mean,
        "mfcc_std_std": mfcc_std_std,
        "mfcc_delta_std": mfcc_delta_std,
        "mfcc_delta2_std": mfcc_delta2_std,
        "mfcc_high_mean": mfcc_high_mean,
        "mfcc_high_std": mfcc_high_std,
        
        # Pitch features
        "pitch_std": pitch_std,
        "pitch_range": pitch_range,
        "pitch_jitter": pitch_jitter,
        "yin_std": yin_std,
        
        # Spectral features
        "centroid_mean": centroid_mean,
        "centroid_std": centroid_std,
        "bandwidth_mean": bandwidth_mean,
        "bandwidth_std": bandwidth_std,
        "rolloff_mean": rolloff_mean,
        "rolloff_std": rolloff_std,
        "flatness_mean": flatness_mean,
        "flatness_std": flatness_std,
        "contrast_mean": contrast_mean,
        "contrast_std": contrast_std,
        
        # High-frequency features
        "hf_energy_mean": hf_energy_mean,
        "hf_energy_std": hf_energy_std,
        "hf_variation": hf_variation,
        "vhf_energy": vhf_energy,
        "spectral_slope": spectral_slope,
        
        # Temporal features
        "rms_std": rms_std,
        "rms_mean": rms_mean,
        "zcr_std": zcr_std,
        "zcr_mean": zcr_mean,
        "dynamic_range": dynamic_range,
        "energy_entropy": energy_entropy,
        
        # Artifact features
        "flux_std": flux_std,
        "flux_mean": flux_mean,
        "spec_irregularity": spec_irregularity,
        "avg_formant_width": avg_formant_width,
        "formant_width_std": formant_width_std,
        "phase_consistency": phase_consistency,
        "low_mid_ratio": low_mid_ratio,
        "mid_high_ratio": mid_high_ratio,
    }


# ─────────────────────────────────────────────────────────────
#  AGGRESSIVE AI CLASSIFIER (Tuned for ElevenLabs)
# ─────────────────────────────────────────────────────────────

def classify_audio(features: dict) -> tuple:
    """Fixed classification with correct verdict logic and safe key access"""
    
    ai_score = 0
    human_score = 0
    details = []
    
    # Safely get features with defaults
    pitch_std = features.get("pitch_std", 0)
    mfcc_var = features.get("mfcc_variance", 0)
    dyn_range = features.get("dynamic_range", 0)
    flatness = features.get("flatness_mean", 0)
    rms_std = features.get("rms_std", 0)
    hf_energy = features.get("hf_energy_mean", 0)
    hf_var = features.get("hf_variation", 0)
    flux_std = features.get("flux_std", 0)
    entropy = features.get("energy_entropy", 0)
    delta_var = features.get("mfcc_delta_std", 0)
    zcr_std = features.get("zcr_std", 0)
    irregularity = features.get("spec_irregularity", 0)
    jitter = features.get("pitch_jitter", 0)
    
    # 1. PITCH VARIATION
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
    
    # 2. MFCC CONSISTENCY
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
    if jitter > 0:
        if jitter < 0.3:
            ai_score += 8
            details.append(f"⚠️ Very low pitch jitter ({jitter:.2f}) - AI")
        elif jitter > 0.8:
            human_score += 6
    
    # Calculate percentages
    total_score = ai_score + human_score
    if total_score > 0:
        ai_percentage = (ai_score / total_score) * 100
        human_percentage = (human_score / total_score) * 100
    else:
        ai_percentage = 50
        human_percentage = 50
    
    # ✅ FIXED VERDICT LOGIC - Higher percentage determines verdict
    if ai_percentage > human_percentage:
        # AI characteristics dominate - FAKE
        verdict = "FAKE"
        confidence = min(0.95, 0.5 + (ai_percentage - 50) / 100)
        prob_fake = round(confidence, 4)
        prob_real = round(1 - prob_fake, 4)
    else:
        # Human characteristics dominate - REAL
        verdict = "REAL"
        confidence = min(0.95, 0.5 + (human_percentage - 50) / 100)
        prob_real = round(confidence, 4)
        prob_fake = round(1 - prob_real, 4)
    
    # Log detailed analysis
    logger.info(f"=== AI Detection Results ===")
    logger.info(f"AI Score: {ai_score}, Human Score: {human_score}")
    logger.info(f"AI: {ai_percentage:.1f}% | Human: {human_percentage:.1f}%")
    logger.info(f"Verdict: {verdict} (Real: {prob_real:.1%}, AI: {prob_fake:.1%})")
    for detail in details[:5]:
        logger.info(f"  {detail}")
    
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
            "method": "ai_optimized_v3",
            "features": {
                "pitch_variation": f"{features['pitch_std']:.1f}",
                "mfcc_variance": round(features['mfcc_variance'], 2),
                "energy_variation": round(features['rms_std'], 4),
                "dynamic_range": f"{features['dynamic_range']:.1f}",
                "spectral_flatness": round(features['flatness_mean'], 5),
            },
            "indicators": indicators
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    
    