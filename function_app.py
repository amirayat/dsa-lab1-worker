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

app = func.FunctionApp()


# ---------- Config from env vars ----------
def _env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise ValueError(f"Missing environment variable: {name}")
    return v


async def process_one_message() -> None:
    """Receive one message from Service Bus, process it, update DB."""
    # Read env vars inside the function so Azure picks up latest values
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
                    body = json.loads(str(msg))
                    image_id = body["image_id"]
                    blob_url = body["blob_url"]
                    logging.info(f"Processing image_id={image_id}, blob_url={blob_url}")

                    # 1. Set state to Sending
                    async with pool.acquire() as conn:
                        await conn.execute(
                            "UPDATE images SET state = $2 WHERE id = $1",
                            image_id, "q2 Sending",
                        )

                    try:
                        # 2. Download image from Blob Storage
                        blob_service = BlobServiceClient.from_connection_string(BLOB_CONN_STR)
                        async with blob_service:
                            blob_name = blob_url.split("/")[-1]
                            blob_client = blob_service.get_blob_client(
                                container=BLOB_CONTAINER, blob=blob_name
                            )
                            download = await blob_client.download_blob()
                            image_bytes = await download.readall()

                        # 3. Call Azure Face
                        face_client = FaceClient(
                            FACE_ENDPOINT, AzureKeyCredential(FACE_KEY)
                        )
                        async with face_client:
                            result = await face_client.detect(
                                image_content=image_bytes,
                                detection_model="detection_03",
                                return_face_attributes=[
                                    "glasses", "mask", "headpose",
                                    "blur", "exposure", "qualityforrecognition",
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

                            # Serialize face data
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

                        # 4. Update DB: state → ResultsReady, save result
                        async with pool.acquire() as conn:
                            await conn.execute(
                                """
                                UPDATE images
                                SET state = $2, result_json = $3::jsonb
                                WHERE id = $1
                                """,
                                image_id, "q7 ResultsReady", result_json,
                            )

                            # 5. Increment request counter
                            await conn.execute(
                                """
                                UPDATE requests
                                SET completed = completed + 1,
                                    status = CASE
                                        WHEN completed + 1 >= total THEN 'done'
                                        ELSE status
                                    END
                                WHERE id = (SELECT request_id FROM images WHERE id = $1)
                                """,
                                image_id,
                            )

                        # 6. Complete the message (remove from queue)
                        await receiver.complete_message(msg)
                        logging.info(f"✅ Completed image_id={image_id} → {new_state}")

                    except Exception as e:
                        logging.error(f"❌ Error processing {image_id}: {e}")
                        # Abandon → message returns to queue for retry
                        await receiver.abandon_message(msg)

    finally:
        await pool.close()


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