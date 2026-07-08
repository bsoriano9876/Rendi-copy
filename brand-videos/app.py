from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from pathlib import Path
import asyncio, httpx, os, uuid, subprocess, boto3
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

API_SECRET_KEY  = os.getenv("API_SECRET_KEY", "changeme")
DO_SPACES_KEY    = os.getenv("DO_SPACES_KEY")
DO_SPACES_SECRET = os.getenv("DO_SPACES_SECRET")
DO_SPACES_BUCKET = os.getenv("DO_SPACES_BUCKET", "nightowl-bucket")
DO_SPACES_REGION = os.getenv("DO_SPACES_REGION", "sfo3")

class ProcessRequest(BaseModel):
    video_url:    str
    candidate_id: str
    callback_url: str
    background:   str = "meeting_dark"

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/process")
async def process_video(
    request: ProcessRequest,
    x_api_key: str = Header(None)
):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    job_id = str(uuid.uuid4())[:8]
    asyncio.create_task(run_brand_job(job_id, request))
    return {"job_id": job_id, "status": "queued"}

# ── Background job ────────────────────────────────────────────────────────────
async def run_brand_job(job_id: str, request: ProcessRequest):
    urls_file  = f"/tmp/urls_{job_id}.txt"
    output_dir = f"/tmp/output_{job_id}"

    Path(urls_file).write_text(request.video_url)
    Path(output_dir).mkdir(exist_ok=True)

    script_dir = Path(__file__).parent
    bg_filename = "Meeting Background 2.png" if request.background == "meeting_light" else "Meeting Background 1.png"
    bg_image   = script_dir / bg_filename
    logo       = script_dir / "test_logo.png"

    proc = subprocess.run([
        "python3", str(script_dir / "brand_videos.py"),
        "--urls",       urls_file,
        "--logo",       str(logo),
        "--output-dir", output_dir,
        "--bg-image",   str(bg_image),
        "--segmenter",  "mediapipe",
        "--logo-scale", "0.001",
        "--upscale",    "0"
    ], text=True)

    branded_url = None
    if proc.returncode == 0:
        mp4_files = list(Path(output_dir).glob("*.mp4"))
        if mp4_files:
            branded_url = upload_to_spaces(mp4_files[0], request.candidate_id, job_id)

    # Cleanup temp files
    Path(urls_file).unlink(missing_ok=True)
    for f in Path(output_dir).glob("*"):
        f.unlink()
    Path(output_dir).rmdir()

    # Callback to n8n
    async with httpx.AsyncClient() as client:
        await client.post(request.callback_url, json={
            "candidate_id":      request.candidate_id,
            "job_id":            job_id,
            "branded_video_url": branded_url,
            "status":            "success" if branded_url else "failed",
            "error":             "Check Render logs for details" if not branded_url else None
        }, timeout=30)

# ── Upload to DigitalOcean Spaces ─────────────────────────────────────────────
def upload_to_spaces(file_path: Path, candidate_id: str, job_id: str) -> str:
    s3 = boto3.client(
        "s3",
        endpoint_url=f"https://{DO_SPACES_REGION}.digitaloceanspaces.com",
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET,
    )
    key = f"moodle_test_center/branded_videos/{candidate_id}_{job_id}.mp4"
    s3.upload_file(
        str(file_path),
        DO_SPACES_BUCKET,
        key,
        ExtraArgs={"ACL": "public-read", "ContentType": "video/mp4"}
    )
    return f"https://{DO_SPACES_BUCKET}.{DO_SPACES_REGION}.digitaloceanspaces.com/{key}"
