# dsa-lab1-worker

Azure Function worker (Python 3.12, timer every 30s) for the DSA Lab 1 distributed image pipeline.

---

## What It Does

Every 30 seconds, the worker:

1. Receives one message from Azure Service Bus (image-processing-queue).
2. Parses the Event Grid event to get the blob URL.
3. Downloads the image from Azure Blob Storage.
4. Validates the image format (JPEG/PNG/GIF/BMP/WebP).
5. Calls Azure AI Face (glasses, mask, headpose, blur, exposure, quality).
6. Calls Azure Translator to translate labels to Romanian (_ro fields).
7. Updates PostgreSQL (state='q7 ResultsReady', result_json).
8. Completes the Service Bus message.

---

## Pipeline

Client -> FastAPI/gRPC -> Blob Storage -> Event Grid -> Service Bus
                                                          |
                                                          v
                                                THIS WORKER (30s)
                                                          |
                                                          v
                                            Azure Face + Translator
                                                          |
                                                          v
                                                     PostgreSQL

---

## FSM States

| State | Meaning |
|-------|---------|
| q1 Queued | Set by the API on submission |
| q2 Sending | Set by the worker before Face call |
| q4 SingleFaceDetected | One face found |
| q5 MultipleFacesDetected | Two or more faces |
| q6 NoFaceDetected | Zero faces (valid outcome) |
| q7 ResultsReady | Result saved |
| q9 Retry | Transient error — will retry |
| q10 Failed | Permanent error |

---

## Project Files

dsa-lab1-worker/
  function_app.py       entry point
  host.json             Function runtime config
  requirements.txt      Python dependencies
  local.settings.json   local env vars (gitignored)
  .github/workflows/    GitHub Actions deployment

---

## Environment Variables

Runtime:
- AzureWebJobsStorage — attached storage account connection string

PostgreSQL:
- POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

Service Bus:
- SERVICE_BUS_CONNECTION_STRING, SERVICE_BUS_QUEUE_NAME

Blob Storage:
- BLOB_CONNECTION_STRING, BLOB_CONTAINER_NAME

Azure AI Face:
- AZURE_FACE_KEY, AZURE_FACE_ENDPOINT

Azure Translator:
- TRANSLATOR_KEY, TRANSLATOR_REGION, TRANSLATOR_TARGET_LANG

---

## Local Development

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
func start

Expected output:

Functions:
    worker_timer: timerTrigger

Timer fired — checking queue for one message.
No messages in queue.

For local runs, put your secrets in local.settings.json (gitignored).

---

## Deploy

Automatic via GitHub Actions on push to main.

Manual:

func azure functionapp publish dsa-lab1-worker-v2

---

## Logs (Azure Portal -> Function -> Logs)

Successful run:

Event Grid -> image_id=...
Detected image format: jpeg (49411 bytes)
Translated 5 labels to 'ro'
Completed image_id=... -> q4 SingleFaceDetected

---

## Error Handling

| Error | Behavior |
|-------|----------|
| Parse error | Dead-lettered |
| Unsupported format (AVIF) | q10 Failed, message completed |
| Face 4xx | q10 Failed, message completed |
| Transient error | Abandoned -> retried (up to 10x) |
| Max retries | Dead-letter queue |

---

## Related Repositories

- dsa-lab1-api — FastAPI + REST API (App Service)
- dsa-lab1-grpc — gRPC server + client (Part 3)

---

## FSM Specification (Part 0)

M = (Q, Sigma, delta, q0, F)

q0 = Idle
F  = {Delivered, Failed, Cancelled}

The images.state column stores the current FSM state of every image.

---

Laboratory work — Kaunas University of Technology, 2026.