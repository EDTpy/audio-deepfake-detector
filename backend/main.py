# flake8: noqa
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import librosa
import io
import logging
from scipy import stats
from scipy.signal import find_peaks

app = FastAPI(title="Audio Deepfake Detector")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any frontend to connect
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  ADVANCED FEATURE EXTRACTION FOR AI VOICE DETECTION
#  Specifically designed to catch ElevenLabs and modern TTS
# ─────────────────────────────────────────────────────────────

def extract_advanced_features(audio: np.ndarray, sr: int) -> dict:
    """Extract features that reveal AI-generated artifacts"""
    
    # Ensure minimum length
    min_samples = sr * 3
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))
    
    # Trim silence
    audio_trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(audio_trimmed) > sr * 0.5:
        audio = audio_trimmed
    
    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak
    
    hop_length = 256
    n_fft = 2048
    
    # ─────────────────────────────────────────────────────────
    # 1. MFCC FEATURES (Key for detecting synthetic artifacts)
    # ─────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20, hop_length=hop_length)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # MFCC statistics - AI voices have less variation
    mfcc_std_mean = float(np.std(mfcc, axis=1).mean())
    mfcc_std_std = float(np.std(mfcc, axis=1).std())
    mfcc_delta_std = float(np.std(mfcc_delta))
    mfcc_delta2_std = float(np.std(mfcc_delta2))
    
    # MFCC high-frequency coefficients (reveal synthesis artifacts)
    mfcc_high_mean = float(mfcc[10:].mean())
    mfcc_high_std = float(mfcc[10:].std())
    
    # ─────────────────────────────────────────────────────────
    # 2. PITCH FEATURES (AI voices are too stable)
    # ─────────────────────────────────────────────────────────
    try:
        # Use multiple pitch detection methods
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
    # 3. SPECTRAL FEATURES (AI has unnatural spectral patterns)
    # ─────────────────────────────────────────────────────────
    stft = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # Spectral centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)[0]
    centroid_std = float(np.std(spectral_centroids))
    centroid_mean = float(np.mean(spectral_centroids))
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, hop_length=hop_length)[0]
    bandwidth_std = float(np.std(spectral_bandwidth))
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, hop_length=hop_length)[0]
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
    # 4. HIGH-FREQUENCY ANALYSIS (AI often has unnatural HF)
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
    # 5. TEMPORAL FEATURES (AI has unnatural smoothness)
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
    # 6. ARTIFACT DETECTION (Specific to TTS/vocoders)
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
        from scipy.signal import find_peaks
        formant_widths = []
        for frame in stft[:, :50].T:  # Check first 50 frames
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
        "centroid_std": centroid_std,
        "bandwidth_std": bandwidth_std,
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
        "zcr_std": zcr_std,
        "dynamic_range": dynamic_range,
        "energy_entropy": energy_entropy,
        
        # Artifact features
        "flux_std": flux_std,
        "spec_irregularity": spec_irregularity,
        "avg_formant_width": avg_formant_width,
        "formant_width_std": formant_width_std,
        "phase_consistency": phase_consistency,
        "low_mid_ratio": low_mid_ratio,
        "mid_high_ratio": mid_high_ratio,
    }


def classify_audio(features: dict) -> tuple:
    """Aggressive classification specifically tuned for modern AI voices"""
    
    ai_score = 0
    human_score = 0
    details = []
    
    # 1. PITCH VARIATION (HIGHEST WEIGHT - Most discriminative)
    pitch_std = features["pitch_std"]
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
    
    # 2. MFCC CONSISTENCY (AI has less variation)
    mfcc_var = features["mfcc_variance"]
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
    
    # 3. DYNAMIC RANGE (AI has compressed dynamics)
    dyn_range = features["dynamic_range"]
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
    
    # 4. SPECTRAL FLATNESS (AI too clean or unnatural)
    flatness = features["flatness_mean"]
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
    
    # 5. ENERGY VARIATION (AI has less energy fluctuation)
    rms_std = features["rms_std"]
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
    
    # 6. HIGH-FREQUENCY CONTENT (AI often has unnatural HF)
    hf_energy = features["hf_energy_mean"]
    hf_var = features["hf_variation"]
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
    
    # 7. SPECTRAL FLUX (AI has smoother transitions)
    flux_std = features["flux_std"]
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
    
    # 8. ENERGY ENTROPY (AI has more predictable patterns)
    entropy = features["energy_entropy"]
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
    
    # 9. MFCC DELTA VARIATION (temporal dynamics)
    delta_var = features["mfcc_delta_std"]
    if delta_var < 3.5:
        ai_score += 12
        details.append(f"⚠️ Very smooth MFCC transitions ({delta_var:.1f}) - AI")
    elif delta_var < 5.5:
        ai_score += 5
    elif delta_var > 9:
        human_score += 8
        details.append(f"✅ Natural MFCC dynamics ({delta_var:.1f})")
    
    # 10. ZCR VARIATION (consonant transitions)
    zcr_std = features["zcr_std"]
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
    irregularity = features["spec_irregularity"]
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
    jitter = features["pitch_jitter"]
    if jitter > 0 and jitter < 0.3:
        ai_score += 8
        details.append(f"⚠️ Very low pitch jitter ({jitter:.2f}) - AI")
    elif jitter > 0.8:
        human_score += 6
    
    # Calculate final percentages
    total_score = ai_score + human_score
    if total_score > 0:
        ai_percentage = (ai_score / total_score) * 100
        human_percentage = (human_score / total_score) * 100
    else:
        ai_percentage = 50
        human_percentage = 50
    
    # Determine verdict and confidences based on which percentage is higher
    if ai_percentage > human_percentage:
        verdict = "FAKE"
        # Scale confidence: 50% AI = 0.5, 100% AI = 0.95
        confidence = min(0.95, 0.5 + (ai_percentage - 50) / 100)
        prob_fake = round(confidence, 4)
        prob_real = round(1 - prob_fake, 4)
    else:
        verdict = "REAL"
        # Scale confidence: 50% human = 0.5, 100% human = 0.95
        confidence = min(0.95, 0.5 + (human_percentage - 50) / 100)
        prob_real = round(confidence, 4)
        prob_fake = round(1 - prob_real, 4)
    
    # Ensure probabilities are not exactly 0.5 for decisive verdict
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
    for detail in details[:6]:
        logger.info(f"  {detail}")
    
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
    
    # Extract features and classify
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
            "method": "advanced_heuristic",
            "features": {
                "pitch_variation": f"{features['pitch_std']:.1f}",
                "mfcc_variation": round(features['mfcc_std_mean'], 2),
                "energy_variation": round(features['rms_std'], 4),
                "dynamic_range": f"{features['dynamic_range']:.1f}",
                "spectral_flatness": round(features['flatness_mean'], 5),
            },
            "indicators": indicators[:4]  # Return top indicators
        }
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        