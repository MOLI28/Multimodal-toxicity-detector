import os
import re
import cv2
import json
import uuid
import numpy as np
import logging
import whisper
import base64
import time
import urllib.request
import mediapipe as mp
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from transformers import pipeline
import google.generativeai as genai

# --- GOOGLE MEDIAPIPE BYPASS ---
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import mediapipe_c_bindings as c_bindings
original_load = c_bindings.load_raw_library
def safe_load_library(*args, **kwargs):
    try: return original_load(*args, **kwargs)
    except AttributeError as e:
        if "free" in str(e).lower(): return c_bindings._shared_lib
        raise
c_bindings.load_raw_library = safe_load_library

# --- DEEP SCAN IMPORTS ---
from audio_pipeline import AudioPipeline
from video_pipeline import run_video_pipeline
from aligner import fuse_modalities
from rag_engine import run_policy_rag
from merger import merge_audio_to_video
from format_converter import convert_to_web_format

# --- API & APP CONFIG ---
genai.configure(api_key="PASTE_YOUR_GEMINI_API_KEY_HERE")
llm_jury = genai.GenerativeModel('gemini-1.5-flash')
logging.getLogger("transformers").setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', max_http_buffer_size=10000000)

user_strikes, user_memory, user_last_strike = {}, {}, {}

# --- LIGHTWEIGHT LIVE MODELS ---
print("Moli AI [OMNI-MODAL SOC] Booting up...")
live_whisper_model = whisper.load_model("tiny")
toxicity_analyzer = pipeline("text-classification", model="martin-ha/toxic-comment-model")

def get_toxicity_score(text):
    """Safely extracts the toxicity probability without punishing safe text."""
    if not text: return 0.0
    result = toxicity_analyzer(text)[0]
    label = result['label'].lower()
    score = float(result['score'])
    
    # If the AI says it is NON-toxic, we reverse the score to practically zero
    if 'non' in label or 'safe' in label:
        return 1.0 - score
    
    # If it IS toxic, return the high threat score
    return score

def download_model(url, filename):
    if not os.path.exists(filename): urllib.request.urlretrieve(url, filename)

download_model("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 'hand_landmarker.task')
download_model("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", 'face_landmarker.task')
download_model("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task", 'gesture_recognizer.task')

face_base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
face_options = vision.FaceLandmarkerOptions(base_options=face_base_options, output_face_blendshapes=True, num_faces=3)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

gesture_base_options = mp_python.BaseOptions(model_asset_path='gesture_recognizer.task')
gesture_options = vision.GestureRecognizerOptions(base_options=gesture_base_options)
gesture_recognizer = vision.GestureRecognizer.create_from_options(gesture_options)

BLOCK_LIST = ["idiot", "stupid", "dumb", "loser", "shut up", "trash", "bastard", "asshole", "bitch", "fucker", "dick", "wtf", "pagal", "chutiya", "saala", "kutta", "कमीना", "पागल"]
TOXIC_EMOJIS = ["🖕", "🔪", "🔫", "🩸", "☠️", "🤬", "👿", "👊"]

# --- LIVE VISION ENGINE ---
def analyze_visuals(img):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    gesture_status, facial_status, faces_detected = "None", "Neutral", 0
    target_box = None

    try:
        # 1. HAND GESTURE LOGIC (WITH UNIVERSAL CENTER SHIELD)
        hand_result = gesture_recognizer.recognize(mp_image)
        
        if hand_result.hand_landmarks:
            # Calculate the center of the hand
            avg_y = sum([lm.y for lm in hand_result.hand_landmarks[0]]) / 21
            avg_x = sum([lm.x for lm in hand_result.hand_landmarks[0]]) / 21
            
            is_in_safe_zone = (0.25 < avg_x < 0.75) and (avg_y < 0.65)
            
            if not is_in_safe_zone:
                # Part A: Main AI Check
                if hand_result.gestures and hand_result.gestures[0][0].score > 0.7:
                    cat = hand_result.gestures[0][0].category_name
                    if cat == "Closed_Fist": gesture_status = "Aggressive Fist Detected"
                    elif cat == "Open_Palm": gesture_status = "Stop Gesture Detected"

                # Part B: Backup Manual Check
                if gesture_status == "None":
                    for hl in hand_result.hand_landmarks:
                        # Check if fingers are extended
                        ext = [hl[tip].y < hl[pip].y for tip, pip in zip([8,12,16,20], [6,10,14,18])]
                        if ext[1] and not any([ext[0], ext[2], ext[3]]): 
                            gesture_status = "Offensive Gesture Detected"
                        elif not any(ext): 
                            gesture_status = "Aggressive Fist Detected"
    except Exception: pass

    try:
        # 2. FACIAL EXPRESSION & TRACKING LOGIC
        # 2. FACIAL EXPRESSION & TRACKING LOGIC (GUARANTEED DEMO MODE)
        face_result = face_detector.detect(mp_image)
        if face_result.face_landmarks:
            faces_detected = len(face_result.face_landmarks)
            
            # Simplified Logic: If you open your mouth wide or squint your eyes,
            # it triggers instantly regardless of complex blendshape scores.
            # We are now looking at the face landmarker directly.
            landmarks = face_result.face_landmarks[0]
            mouth_dist = abs(landmarks[13].y - landmarks[14].y) # Distance between lips
            
            if mouth_dist > 0.05: # Mouth is open
                facial_status = "Raging / Visual Shouting"
            else:
                facial_status = "Neutral"
        else:
            facial_status = "Neutral"
    except Exception: pass

    return gesture_status, facial_status, faces_detected, target_box

def consult_llm_jury(chat_history):
    prompt = f"Moderator Log: {chat_history}\nIs this malicious harassment or safe banter? Reply exactly with one word: 'HARMFUL' or 'HARMLESS'."
    try: return llm_jury.generate_content(prompt).text.strip().upper()
    except: return "HARMFUL" 

# --- UPGRADED 3-VARIABLE FUSION ENGINE ---
def process_violation(user_id, role, text, visual_status, faces_detected, bert_score):
    global user_strikes, user_memory, user_last_strike
    is_violation, reason_detail = False, ""
    if role in ["TEACHER", "ADMIN"]: return "NONE", "Authority Exempt", bert_score
    
    current_time = time.time()
    cooldown_active = (user_id in user_last_strike and (current_time - user_last_strike[user_id]) < 4.0)
    
    # 1. Advanced Catching: Find and SAVE the EXACT word or emoji
    matched_word = None
    matched_emoji = None

    try:
        from roberta_detector import TOXIC_TAXONOMY, collapse_duplicates
        clean_text = collapse_duplicates(text)
        for data in TOXIC_TAXONOMY.values():
            for bw in data["words"]:
                if bw in clean_text:
                    matched_word = bw
                    break
            if matched_word: break
    except ImportError:
        for bw in BLOCK_LIST:
            if bw in text.lower():
                matched_word = bw
                break

    for emoji in TOXIC_EMOJIS:
        if emoji in text:
            matched_emoji = emoji
            break

    contains_hindi_script = any('\u0900' <= char <= '\u097F' for char in text)

    # 2. Perfect Reason Generation
    if matched_word:
        bert_score = 1.0  # Force the UI red!
        is_violation = True
        reason_detail = f"Vocabulary Violation: Blocked word '{matched_word}'"
    elif matched_emoji:
        bert_score = 1.0  # Force the UI red!
        is_violation = True
        reason_detail = f"Emoji Violation: Blocked symbol '{matched_emoji}'"
    else:
        fused_threat_score = bert_score
        fusion_reasons = []
        if bert_score > 0.2: fusion_reasons.append(f"Toxic-BERT({bert_score:.2f})")
        # 1. IMMEDIATE VIOLATIONS (Visual Demo Force-Trigger)
        is_visual_violation = (visual_status == "Raging / Visual Shouting")
        if visual_status in ["Aggressive Fist Detected", "Offensive Gesture Detected"]:
            fused_threat_score += 0.6; fusion_reasons.append(f"Gesture({visual_status})")
        elif visual_status in ["Raging / Visual Shouting", "Aggressive / Mocking", "Intimidating Glare"]:
            fused_threat_score += 0.4; fusion_reasons.append(f"Expression({visual_status})")
        if faces_detected > 1:
            fused_threat_score += 0.5; fusion_reasons.append(f"SecurityBreach({faces_detected} Faces)")
        # 2. ANALYTICAL VIOLATIONS (The "Jury" Logic)
        needs_jury = (fused_threat_score >= 0.60 or contains_hindi_script)
        # FINAL GATEKEEPER
        if is_visual_violation or (needs_jury and "HARMFUL" in consult_llm_jury(user_memory.get(user_id, []))):
            is_violation = True
            reason_detail = f"Fused Threat Confirmed: {' + '.join(fusion_reasons) if fusion_reasons else 'Context Check'}"
            bert_score = max(bert_score, 0.85)

    if is_violation and not cooldown_active:
        user_last_strike[user_id] = current_time 
        user_strikes[user_id] = user_strikes.get(user_id, 0) + 1
        recent_reason = reason_detail
    else:
        recent_reason = reason_detail if is_violation else "Monitoring Active"

    strikes = user_strikes.get(user_id, 0)
    
    # 3. Returning THREE variables with dynamic reporting
    if strikes == 0: 
        return "NONE", "System operating normally.", bert_score
    elif strikes <= 2: 
        return "WARN", f"Strike {strikes}: {recent_reason}", bert_score
    elif 3 <= strikes <= 4: 
        return "MUTE", f"System Lockout: {recent_reason}", bert_score
    else: 
        return "BLOCK", f"Maximum strikes exceeded: {recent_reason}", bert_score

# --- SOCKETS ---
@socketio.on('connect')
def handle_connect(): print("[CONN]: Active WebSocket frontend connected.")

@socketio.on('stream_frame')
def handle_live_frame(data):
    socketio.sleep(0.01)
    user_id = data.get('user_id', 'student_live')
    try:
        encoded_frame = data['frame'].split(',')[1] if ',' in data['frame'] else data['frame']
        img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded_frame), np.uint8), cv2.IMREAD_COLOR)
        visual_status, facial_status, faces_detected, target_box = analyze_visuals(img)
    except:
        visual_status, facial_status, faces_detected, target_box = "None", "Neutral", 1, None

    # Unpack 3 variables!
    if facial_status == "Raging / Visual Shouting":
        action = "WARN"
        reason = "Behavioral Violation: Raging Face Detected"
        bert_score = 0.9
        # Manually increment strike for demo effect if needed
        user_strikes[user_id] = user_strikes.get(user_id, 0) + 1 
    else:
        # Standard analytical path
        action, reason, bert_score = process_violation(user_id, "STUDENT", "", visual_status, faces_detected, 0.0)
    
    emit('moderation_alert', {
        "user_id": user_id, 
        "strikes": user_strikes.get(user_id,0), 
        "visual_status": visual_status, 
        "facial_status": facial_status, 
        "faces": faces_detected, 
        "action": action, 
        "reason": reason,
        "target_box": target_box 
    })

@socketio.on('send_chat')
def handle_live_chat(data):
    socketio.sleep(0.01)
    try:
        user_id, text = data.get('user_id', 'student_live'), data.get('text', '')
        if user_id not in user_memory: user_memory[user_id] = []
        if text.strip():
            user_memory[user_id].append(text)
            if len(user_memory[user_id]) > 5: user_memory[user_id].pop(0)

        bert_score = get_toxicity_score(text)
        
        # Unpack 3 variables and capture updated_bert_score!
        action, reason, updated_bert_score = process_violation(user_id, "STUDENT", text, "None", 1, bert_score)
        
        emit('chat_verdict', {
            "user_id": user_id, "strikes": user_strikes.get(user_id,0), 
            "text": text, "action": action, "reason": reason, 
            "bert_confidence": updated_bert_score 
        })  
    except Exception as e: print(f"Chat Error: {e}")

@socketio.on('webcam_frame')
def handle_webcam_frame(data):
    try:
        img_data = data['image'].split(",")[1]
        img_bytes = base64.b64decode(img_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        recognition_result = gesture_recognizer.recognize(mp_image)

        if recognition_result.gestures:
            top_gesture = recognition_result.gestures[0][0].category_name
            print(f"[GESTURE DETECTED]: {top_gesture}")
            emit('gesture_response', {'gesture': top_gesture})
        else:
            emit('gesture_response', {'gesture': 'None'})

    except Exception as e:
        print(f"Error processing frame: {e}")

# --- REST ENDPOINTS ---
@app.route('/')
def home(): return send_file('index.html')

@app.route('/moderate-all', methods=['POST'])
def moderate_all():
    tmp_media_path = None
    try:
        user_id = request.form.get('user_id', 'student_1')
        if user_id not in user_memory: user_memory[user_id] = []
        
        visual_status, facial_status = request.form.get('visual_status', 'None'), request.form.get('facial_status', 'Neutral')
        faces_detected = int(request.form.get('faces_detected', 1))
        transcribed_text, bert_score = "", 0.0
                
        if 'file' in request.files:
            media_file = request.files['file']
            tmp_media_path = f"temp_media.{media_file.filename.split('.')[-1].lower()}"
            media_file.save(tmp_media_path)
            transcribed_text = live_whisper_model.transcribe(tmp_media_path, fp16=False, initial_prompt="Audio in Hinglish, English, Hindi")['text']
        
        if transcribed_text.strip():
            user_memory[user_id].append(transcribed_text)
            bert_score = get_toxicity_score(transcribed_text)
            
        # Unpack 3 variables here too!
        action, reason, updated_bert_score = process_violation(user_id, "STUDENT", transcribed_text, visual_status, faces_detected, bert_score)
        if tmp_media_path and os.path.exists(tmp_media_path): os.remove(tmp_media_path)
        
        return {"status": "success", "strikes": user_strikes.get(user_id,0), "action": action, "reason": reason, "final_text": transcribed_text, "visual_status": visual_status, "facial_status": facial_status, "faces": faces_detected, "bert_confidence": updated_bert_score}

    except Exception as e:
        if tmp_media_path and os.path.exists(tmp_media_path): os.remove(tmp_media_path)
        return {"status": "error", "reason": str(e)}, 500

# ==============================================================================
# OMNI-MODAL BATCH PIPELINE (RAG + RT-DETR + AUDIO FUSION)
# ==============================================================================
offline_audio_pipeline = AudioPipeline()

@app.route('/deep-scan', methods=['POST'])
def deep_scan_video():
    """Heavy batch processing route orchestrating all external files."""
    if 'file' not in request.files:
        return {"status": "error", "reason": "No file uploaded"}, 400
        
    uploaded_file = request.files['file']
    job_id = f"job_{str(uuid.uuid4())[:8]}"
    work_dir = os.path.join("processing_jobs", job_id)
    os.makedirs(work_dir, exist_ok=True)
    
    input_video_path = os.path.join(work_dir, "input.mp4")
    uploaded_file.save(input_video_path)
    
    try:
        print(f"\n[OMNI-SCAN] Orchestrating heavy pipeline for {job_id}...")
        
        video_out_path = os.path.join(work_dir, "blurred_video.mp4")
        print("[OMNI-SCAN] Running Video Pipeline (CLIP + RT-DETR)...")
        video_results = run_video_pipeline(input_video_path, video_out_path)
        vision_segments = video_results.get("violent_segments", [])
        
        print("[OMNI-SCAN] Running Audio Pipeline (Transcribe + Toxicity + Censor)...")
        audio_results = offline_audio_pipeline.run(input_video_path, work_dir)
        censored_audio_path = audio_results["censored_audio"]
        audio_segments = audio_results.get("audio_events", [])
        
        print("[OMNI-SCAN] Fusing Vision and Audio events...")
        fused_events = fuse_modalities(vision_segments, audio_segments)
        
        evidence_data = {"video_id": job_id, "fused_events": fused_events}
        evidence_path = os.path.join(work_dir, "evidence.json")
        with open(evidence_path, "w") as f:
            json.dump(evidence_data, f)
            
        print("[OMNI-SCAN] Running FAISS RAG against YouTube Policies...")
        rag_output_path = os.path.join(work_dir, "rag_report.json")
        run_policy_rag(evidence_path, rag_output_path)
        
        with open(rag_output_path, "r") as f:
            rag_report = json.load(f)
            
        print("[OMNI-SCAN] Stitching censored media...")
        merged_video_path = os.path.join(work_dir, "merged.mp4")
        merge_audio_to_video(video_out_path, censored_audio_path, merged_video_path)
        
        final_web_video = os.path.join(work_dir, "final_safe.mp4")
        convert_to_web_format(merged_video_path, final_web_video)
        
        print(f"[OMNI-SCAN] Job {job_id} Completed Successfully.")
        return jsonify({"status": "success", "report": rag_report})
        
    except Exception as e:
        print(f"[OMNI-SCAN ERROR]: {e}")
        return jsonify({"status": "error", "reason": str(e)}), 500

if __name__ == '__main__': 
    socketio.run(app, debug=True, port=5000)