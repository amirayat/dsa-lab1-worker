"""
Azure Function Worker — Timer Trigger (every 30 seconds)
Processes one queued image per invocation.

Student: Amir Ayat, Internal Code: F3319, Variant: 7 (Azure Face)
"""
import os
import json
import logging
import asyncio

import azure.functions as func
import asyncpg
from azure.servicebus.aio import ServiceBusClient
from azure.storage.blob.aio import BlobServiceClient
from azure.ai.vision.face.aio import FaceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

# ============================================================
# FunctionApp() must be created at module level.
# ============================================================
app = func.FunctionApp()


# ---------- Helpers ----------

def _env(name: str) -> str:
    """Return an environment variable or raise if missing."""
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing environment variable: {name}")
    return v


def _sniff_image_format(data: bytes) -> str:
    """
    Return the image format based on magic bytes, or 'unknown'.
    Face API supports: JPEG, PNG, GIF, BMP, WebP.
    AVIF is detected but NOT supported.
    """
    if len(data) < 12:
        return "unknown"
    if data[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM":
        return "bmp"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"avif", b"avis"):
            return "avif"
    return "unknown"


# ---------- Worker logic ----------

async def process_one_message() -> None:
    """
    Receive one message from Service Bus, process it, update PostgreSQL.

    FSM states used:
      q1 Queued         — set by the API when the image is submitted
      q2 Sending        — set here just before the Azure Face call
      q4 SingleFaceDetected / q5 MultipleFacesDetected / q6 NoFaceDetected
      q7 ResultsReady   — set after saving result_json
      q10 Failed        — set on permanent errors (unsupported format, Face 4xx)
    """
    POSTGRES_HOST = _env("POSTGRES_HOST")
    POSTGRES_PORT = int(os.environ.get("POSTGRES_PORT", "5432"))
    POSTGRES_DB = _env("POSTGRES_DB")
    POSTGRES_USER = _env("POSTGRES_USER")
    POSTGRES_PASSWORD = _env("POSTGRES_PASSWORD")
    SB_CONN_STR = _env("SERVICE_BUS_CONNECTION_STRING")
    SB_QUEUE = _env("SERVICE_BUS_QUEUE_NAME")
    BLOB_CONN_STR = _env("BLOB_CONNECTION_STRING")
    BLOB_CONTAINER = _env("BLOB_CONTAINER_NAME")
    FACE_KEY = _env("AZURE_FACE_KEY")
    FACE_ENDPOINT = _env("AZURE_FACE_ENDPOINT")

    pool = await asyncpg.create_pool(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        ssl="require",
        min_size=1,
        max_size=2,
    )

    try:
        async with ServiceBusClient.from_connection_string(SB_CONN_STR) as sb:
            receiver = sb.get_queue_receiver(queue_name=SB_QUEUE, max_wait_time=5)

            async with receiver:
                messages = await receiver.receive_messages(
                    max_message_count=1,
                    max_wait_time=5,
                )

                if not messages:
                    logging.info("No messages in queue.")
                    return

                for msg in messages:
                    raw = str(msg)
                    logging.info(f"Received message (first 200 chars): {raw[:200]}")

                    # ---------- Parse the message ----------
                    try:
                        body = json.loads(raw)

                        if body.get("eventType") == "Microsoft.Storage.BlobCreated":
                            blob_url = body["data"]["url"]
                            blob_name = blob_url.split("/")[-1]
                            image_id = blob_name.rsplit(".", 1)[0]
                            logging.info(f"Event Grid → image_id={image_id}")
                        else:
                            image_id = body["image_id"]
                            blob_url = body["blob_url"]
                            logging.info(f"Legacy message → image_id={image_id}")

                    except Exception as parse_err:
                        logging.error(f"Failed to parse message: {parse_err}")
                        await receiver.dead_letter_message(
                            msg,
                            reason="ParseError",
                            error_description=str(parse_err),
                        )
                        continue

                    # ---------- Set state to q2 Sending ----------
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE images SET state = $2 WHERE id = $1",
                            image_id, "q2 Sending",
                        )

                    try:
                        # ---------- Download image from Blob ----------
                        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
                        async with blob_service:
                            blob_name = blob_url.split("/")[-1]
                            blob_client = blob_service.get_blob_client(
                                container=BLOB_CONTAINER, blob=blob_name,
                            )
                            download = await blob_client.download_blob()
                            image_bytes = await download.readall()

                        # ---------- Validate image format via magic bytes ----------
                        fmt = _sniff_image_format(image_bytes)
                        logging.info(
                            f"Detected image format: {fmt} ({len(image_bytes)} bytes)"
                        )

                        if fmt not in ("jpeg", "png", "gif", "bmp", "webp"):
                            # Unsupported → mark as Failed, complete the message
                            async with pool.acquire() as conn:
                                await conn.execute(
                                    "UPDATE images SET state = $2, error = $3 WHERE id = $1",
                                    image_id,
                                    "q10 Failed",
                                    f"Unsupported image format: {fmt}. "
                                    f"Face API supports JPEG/PNG/GIF/BMP/WebP only.",
                                )
                                await conn.execute(
                                    """
                                    UPDATE requests
                                    SET completed = completed + 1,
                                        failed    = failed + 1,
                                        status    = CASE
                                                        WHEN completed + 1 >= total THEN 'done'
                                                        ELSE status
                                                    END
                                    WHERE id = (SELECT request_id FROM images WHERE id = $1)
                                    """,
                                    image_id,
                                )
                            await receiver.complete_message(msg)
                            logging.info(f"⚠️  Skipped image_id={image_id} (format={fmt})")
                            continue

                        # ---------- Call Azure Face ----------
                        face_client = FaceClient(FACE_ENDPOINT, AzureKeyCredential(FACE_KEY))
                        async with face_client:
                            result = await face_client.detect(
                                image_content=image_bytes,
                                detection_model="detection_03",
                                recognition_model="recognition_04",
                                return_face_id=False,
                                return_face_attributes=[
                                    "glasses",
                                    "mask",
                                    "headpose",
                                    "blur",
                                    "exposure",
                                    "qualityforrecognition",
                                ],
                            )
                            faces = list(result)
                            count = len(faces)

                            if count == 0:
                                new_state = "q6 NoFaceDetected"
                            elif count == 1:
                                new_state = "q4 SingleFaceDetected"
                            else:
                                new_state = "q5 MultipleFacesDetected"

                            result_data = []
                            for f in faces:
                                fa = f.face_attributes
                                mask = getattr(fa, "mask", None)
                                result_data.append({
                                    "glasses": getattr(fa, "glasses", None),
                                    "mask": getattr(mask, "type", "noMask") if mask else "noMask",
                                    "mask_confidence": getattr(mask, "confidence", 0.0) if mask else 0.0,
                                    "blur": getattr(getattr(fa, "blur", None), "blur_level", None),
                                    "exposure": getattr(getattr(fa, "exposure", None), "exposure_level", None),
                                    "quality": getattr(fa, "quality_for_recognition", None),
                                })
                            result_json = json.dumps(result_data)

                        # ---------- Update DB ----------
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """
                                UPDATE images
                                SET state = $2, result_json = $3::jsonb
                                WHERE id = $1
                                """,
                                image_id, "q7 ResultsReady", result_json,
                            )
                            await conn.execute(
                                """
                                UPDATE requests
                                SET completed = completed + 1,
                                    status    = CASE
                                                    WHEN completed + 1 >= total THEN 'done'
                                                    ELSE status
                                                END
                                WHERE id = (SELECT request_id FROM images WHERE id = $1)
                                """,
                                image_id,
                            )

                        await receiver.complete_message(msg)
                        logging.info(f"✅ Completed image_id={image_id} → {new_state}")

                    except HttpResponseError as face_err:
                        logging.error(
                            f"❌ Face API error for {image_id}: "
                            f"status={face_err.status_code} msg={face_err.message}"
                        )
                        async with pool.acquire() as conn:
                            await conn.execute(
                                "UPDATE images SET state = $2, error = $3 WHERE id = $1",
                                image_id, "q10 Failed", str(face_err.message),
                            )
                        await receiver.complete_message(msg)

                    except Exception as e:
                        logging.error(f"❌ Transient error processing {image_id}: {e}")
                        await receiver.abandon_message(msg)

    finally:
        await pool.close()


# ---------- Timer trigger ----------

@app.timer_trigger(
    schedule="*/30 * * * * *",
    arg_name="myTimer",
    run_on_startup=False,
    use_monitor=False,
)
def worker_timer(myTimer: func.TimerRequest) -> None:
    """Timer trigger — fires every 30 seconds."""
    if myTimer.past_due:
        logging.info("Timer is past due.")

    logging.info("⏰ Timer fired — checking queue for one message.")
    try:
        asyncio.run(process_one_message())
    except Exception as e:
        logging.error(f"Worker error: {e}")