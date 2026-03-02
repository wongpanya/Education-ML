# Education ML API (Deploy-ready)

Flask API สำหรับงาน **Dropout Risk / Score Prediction / Cluster Assignment / Policy Simulation**

## โครงสร้างโฟลเดอร์

```text
education-ml-api/
├─ app.py
├─ requirements.txt
├─ runtime.txt
├─ Procfile
├─ render.yaml
├─ .gitignore
├─ README.md
├─ models/
│  ├─ model1_dropout.pkl
│  ├─ model2_score_reading.pkl
│  ├─ model2_score_math.pkl
│  ├─ model3_cluster.pkl
│  ├─ scaler_cluster.pkl
│  └─ model_metadata.json
└─ sample_requests/
   ├─ dropout.json
   ├─ score.json
   └─ simulate_policy.json
```

> ตอนนี้ใน repo ตัวอย่างนี้มีแค่ `model_metadata.json` เพื่อให้ push ง่าย — กรุณานำไฟล์ `.pkl` ของคุณใส่ใน `models/` ก่อน deploy จริง

## Run local

```bash
pip install -r requirements.txt
python app.py
```

API จะรันที่ `http://localhost:5000`

## Test endpoints

### Health
```bash
curl http://localhost:5000/health
```

### Predict dropout
```bash
curl -X POST http://localhost:5000/predict/dropout \
  -H "Content-Type: application/json" \
  -d @sample_requests/dropout.json
```

### Predict score
```bash
curl -X POST http://localhost:5000/predict/score \
  -H "Content-Type: application/json" \
  -d @sample_requests/score.json
```

### Simulate policy
```bash
curl -X POST http://localhost:5000/simulate/policy \
  -H "Content-Type: application/json" \
  -d @sample_requests/simulate_policy.json
```

## Deploy on Render

### Option A: ผ่านหน้าเว็บ Render
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT`

### Option B: ใช้ `render.yaml`
เชื่อม GitHub repo แล้วให้ Render อ่าน config อัตโนมัติ

## Notes
- ถ้าเรียกจาก frontend คนละโดเมน (เช่น React/JS) อาจต้องเปิด CORS ใน `app.py`
- อย่า commit ข้อมูลนักเรียนจริง (PII)
- ถ้าไฟล์โมเดลใหญ่ > 100MB ใช้ Git LFS หรือ object storage
