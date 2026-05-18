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
    """
    Classify audio as REAL or FAKE using multi-dimensional analysis
    Specifically tuned to catch ElevenLabs and modern TTS
    """
    
    scores = []
    details = []
    
    # ─────────────────────────────────────────────────────────
    # 1. PITCH VARIATION (HIGH WEIGHT - Most discriminative)
    # AI voices are too stable, humans vary naturally
    # ─────────────────────────────────────────────────────────
    pitch_std = features["pitch_std"]
    if pitch_std > 35:
        score = 1.0  # Very natural human
        detail = "Excellent natural pitch variation"
    elif pitch_std > 22:
        score = 0.7
        detail = "Good natural pitch variation"
    elif pitch_std > 14:
        score = 0.3
        detail = "Moderate pitch variation"
    elif pitch_std > 7:
        score = -0.4  # Suspiciously stable
        detail = "Limited pitch variation (possible AI)"
    elif pitch_std > 0:
        score = -0.8  # Very flat = AI
        detail = "Very flat pitch pattern (AI characteristic)"
    else:
        score = -0.5
        detail = "Could not detect pitch"
    
    scores.append(score * 3.5)
    details.append(f"Pitch variation: {pitch_std:.1f}Hz - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 2. MFCC CONSISTENCY (AI has unnaturally consistent MFCCs)
    # ─────────────────────────────────────────────────────────
    mfcc_std = features["mfcc_std_mean"]
    if mfcc_std > 25:
        score = 1.0
        detail = "Natural MFCC variation"
    elif mfcc_std > 16:
        score = 0.6
        detail = "Good MFCC variation"
    elif mfcc_std > 10:
        score = 0.1
        detail = "Moderate MFCC variation"
    elif mfcc_std > 5:
        score = -0.5
        detail = "Low MFCC variation (possible AI)"
    else:
        score = -0.9
        detail = "Very low MFCC variation (AI characteristic)"
    
    scores.append(score * 2.5)
    details.append(f"MFCC consistency: {mfcc_std:.1f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 3. ENERGY VARIATION (AI has compressed dynamics)
    # ─────────────────────────────────────────────────────────
    rms_std = features["rms_std"]
    if rms_std > 0.09:
        score = 1.0
        detail = "Natural energy variation"
    elif rms_std > 0.06:
        score = 0.6
        detail = "Good energy variation"
    elif rms_std > 0.035:
        score = 0.1
        detail = "Moderate energy variation"
    elif rms_std > 0.015:
        score = -0.5
        detail = "Limited energy variation (possible AI)"
    else:
        score = -0.9
        detail = "Very flat energy (AI characteristic)"
    
    scores.append(score * 2.0)
    details.append(f"Energy variation: {rms_std:.4f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 4. DYNAMIC RANGE (AI has compressed dynamics)
    # ─────────────────────────────────────────────────────────
    dyn_range = features["dynamic_range"]
    if dyn_range > 35:
        score = 1.0
        detail = "Excellent dynamic range"
    elif dyn_range > 25:
        score = 0.7
        detail = "Good dynamic range"
    elif dyn_range > 16:
        score = 0.2
        detail = "Moderate dynamic range"
    elif dyn_range > 8:
        score = -0.5
        detail = "Limited dynamic range (possible AI)"
    else:
        score = -0.9
        detail = "Very compressed dynamics (AI characteristic)"
    
    scores.append(score * 1.5)
    details.append(f"Dynamic range: {dyn_range:.1f}dB - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 5. SPECTRAL FLATNESS (AI too clean or too noisy)
    # ─────────────────────────────────────────────────────────
    flatness = features["flatness_mean"]
    if 0.025 < flatness < 0.09:
        score = 0.8  # Natural noise floor
        detail = "Natural spectral flatness"
    elif 0.015 < flatness < 0.15:
        score = 0.2
        detail = "Acceptable spectral flatness"
    elif flatness < 0.008:
        score = -0.7  # Too clean = AI synthesis
        detail = "Too clean (AI synthesis artifact)"
    elif flatness > 0.2:
        score = -0.5
        detail = "Too noisy (possible processing artifact)"
    else:
        score = 0.0
        detail = "Borderline spectral flatness"
    
    scores.append(score * 1.5)
    details.append(f"Spectral flatness: {flatness:.5f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 6. HIGH-FREQUENCY ANALYSIS (AI has unnatural HF)
    # ─────────────────────────────────────────────────────────
    hf_energy = features["hf_energy_mean"]
    hf_variation = features["hf_variation"]
    
    hf_score = 0
    if hf_energy > 0.05 and hf_variation > 0.01:
        hf_score = 0.6
        detail = "Natural high-frequency content"
    elif hf_energy > 0.02 and hf_variation > 0.005:
        hf_score = 0.1
        detail = "Limited high-frequency variation"
    elif hf_energy < 0.01:
        hf_score = -0.6
        detail = "Missing high frequencies (bandlimited AI)"
    else:
        hf_score = -0.3
        detail = "Unnatural high-frequency pattern"
    
    scores.append(hf_score * 1.2)
    details.append(f"High-frequency: {hf_energy:.4f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 7. SPECTRAL IRREGULARITY (AI too smooth or too rough)
    # ─────────────────────────────────────────────────────────
    irregularity = features["spec_irregularity"]
    if 0.2 < irregularity < 0.6:
        score = 0.7
        detail = "Natural spectral irregularity"
    elif 0.1 < irregularity < 1.0:
        score = 0.2
        detail = "Acceptable spectral pattern"
    elif irregularity < 0.08:
        score = -0.6
        detail = "Too smooth (AI synthesis artifact)"
    elif irregularity > 1.2:
        score = -0.4
        detail = "Too irregular (possible artifact)"
    else:
        score = 0.0
        detail = "Borderline spectral pattern"
    
    scores.append(score * 1.2)
    details.append(f"Spectral irregularity: {irregularity:.3f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 8. MFCC DELTA VARIATION (Temporal dynamics)
    # ─────────────────────────────────────────────────────────
    delta_std = features["mfcc_delta_std"]
    if delta_std > 5.5:
        score = 0.8
        detail = "Natural temporal dynamics"
    elif delta_std > 3.5:
        score = 0.3
        detail = "Moderate temporal dynamics"
    elif delta_std > 2.0:
        score = -0.3
        detail = "Limited temporal dynamics"
    else:
        score = -0.7
        detail = "Very smooth transitions (AI characteristic)"
    
    scores.append(score * 1.2)
    details.append(f"MFCC dynamics: {delta_std:.2f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 9. ZCR VARIATION (Captures natural consonant transitions)
    # ─────────────────────────────────────────────────────────
    zcr_std = features["zcr_std"]
    if zcr_std > 0.08:
        score = 0.8
        detail = "Natural consonant variation"
    elif zcr_std > 0.05:
        score = 0.4
        detail = "Good consonant variation"
    elif zcr_std > 0.03:
        score = -0.1
        detail = "Limited consonant variation"
    elif zcr_std > 0.015:
        score = -0.5
        detail = "Very limited variation (possible AI)"
    else:
        score = -0.8
        detail = "Almost no consonant transitions (AI characteristic)"
    
    scores.append(score * 1.0)
    details.append(f"ZCR variation: {zcr_std:.4f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 10. ENERGY ENTROPY (Predictability of energy patterns)
    # AI has more predictable (lower entropy) energy patterns
    # ─────────────────────────────────────────────────────────
    entropy = features["energy_entropy"]
    if entropy > 6.5:
        score = 0.7
        detail = "Unpredictable energy (natural)"
    elif entropy > 5.0:
        score = 0.3
        detail = "Moderately unpredictable"
    elif entropy > 3.5:
        score = -0.3
        detail = "Somewhat predictable (possible AI)"
    else:
        score = -0.7
        detail = "Very predictable energy (AI characteristic)"
    
    scores.append(score * 1.0)
    details.append(f"Energy entropy: {entropy:.2f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 11. PITCH JITTER (Natural speech has irregular pitch changes)
    # ─────────────────────────────────────────────────────────
    jitter = features["pitch_jitter"]
    if jitter > 1.2:
        score = 0.6
        detail = "Natural pitch jitter"
    elif jitter > 0.6:
        score = 0.2
        detail = "Moderate pitch jitter"
    elif jitter > 0.2:
        score = -0.4
        detail = "Low pitch jitter (possible AI)"
    elif jitter > 0:
        score = -0.7
        detail = "Very low pitch jitter (AI characteristic)"
    else:
        score = 0.0
        detail = "Could not measure pitch jitter"
    
    scores.append(score * 0.8)
    details.append(f"Pitch jitter: {jitter:.2f} - {detail}")
    
    # ─────────────────────────────────────────────────────────
    # 12. FORMANT WIDTH (AI has unnaturally narrow/wide formants)
    # ─────────────────────────────────────────────────────────
    formant_width = features["avg_formant_width"]
    if 5 < formant_width < 15:
        score = 0.5
        detail = "Natural formant structure"
    elif 3 < formant_width < 25:
        score = 0.0
        detail = "Acceptable formant structure"
    elif formant_width > 0:
        score = -0.5
        detail = "Unnatural formant width (possible AI)"
    else:
        score = -0.2
        detail = "Could not analyze formants"
    
    scores.append(score * 0.8)
    details.append(f"Formant width: {formant_width:.1f} - {detail}")
    
    # Calculate weighted score
    total_weight = sum([3.5, 2.5, 2.0, 1.5, 1.5, 1.2, 1.2, 1.2, 1.0, 1.0, 0.8, 0.8])
    weighted_sum = sum(scores)
    norm_score = weighted_sum / total_weight
    
    # Apply non-linear transformation for better separation
    # This makes the decision boundary clearer
    if norm_score > 0:
        prob_human = 0.5 + (norm_score / 2)
    else:
        prob_human = 0.5 / (1 + abs(norm_score))
    
    # Calibrate
    prob_human = np.clip(prob_human, 0.05, 0.95)
    prob_human = round(prob_human, 4)
    prob_fake = round(1.0 - prob_human, 4)
    verdict = "REAL" if prob_human >= 0.5 else "FAKE"
    
    # Log detailed analysis
    logger.info(f"=== Analysis Results ===")
    logger.info(f"Verdict: {verdict} (Human: {prob_human:.2%}, AI: {prob_fake:.2%})")
    logger.info(f"Score: {norm_score:.3f}")
    for detail in details[:5]:  # Show top 5 indicators
        logger.info(f"  {detail}")
    
    return verdict, prob_human, prob_fake, details


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
        