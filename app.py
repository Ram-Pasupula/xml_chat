from fastapi import FastAPI, File, UploadFile, Query
from typing import Union
from process import chat_process
from fastapi.responses import StreamingResponse
import glob
import time
import os
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
app = FastAPI(
    title="CCD Chat API",
    docs_url="/",
    redoc_url="/api/redoc",
    generate_schema=False,
)


@app.post("/ccd")
   async def asr(file: UploadFile = File(...)
):

    try:
        start_time = time.time()
        file_location = f"/tmp/ccd/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
        file_path = os.path.dirname(file_location)
        logger.info(f"{file_path}/{file.filename}")
        chat_process(f"{file_path}/{file.filename}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Execution time: {elapsed_time} seconds")
        logger.info(f"Execution time: {elapsed_time} seconds")
        # logger.info(result)
    except Exception:
        raise Exception(status_code=500, detail='File not able to load')
    else:
        return StreamingResponse(
            result,
            media_type="text/plain",
            headers={
                'Content-Disposition': f'attachment; filename="{file.filename}".txt'
            })
    finally:
        try:
            files = glob.glob(f"/tmp/{file.filename}")
            for f in files:
                os.remove(f)
        except Exception:
            pass
        else:
            logger.info("Successfully deleted temp files")
