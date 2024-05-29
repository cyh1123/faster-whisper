import base64
import io
import json
import os

from typing import Union

import uvicorn

from fastapi import FastAPI
from fastapi_offline import FastAPIOffline

from faster_whisper import WhisperModel

# app对象
app = FastAPI()


# 初始化模型
@app.on_event("startup")
async def init_model():
    global model
    model_size = "models/faster-whisper-large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")


@app.get("/")
async def read_root():
    '''
    返回api列表介绍
    '''
    return {"Hello": "World"}

@app.post("/sasr")
async def sasr(message:dict):
    data = message['data']
    data = base64.b64decode(data)
    data = io.BytesIO(data)
    segments, info = model.transcribe(data, beam_size=5)
    output = ''
    for segment in segments:
        output += segment.text
    return {"response":output}

if __name__ == '__main__':
    # 在调试的时候开源加入一个reload=True的参数，正式启动的时候可以去掉
    uvicorn.run(app="api:app", host="127.0.0.1", port=6605, log_level="info", reload=False)



# ----------------------------------- 调用示例 ----------------------------------- #
# import requests
# URL = 'http://xx.xxx.xxx.63:6605/predict'
# # 这里请注意，data的key，要和我们上面定义方法的形参名字和数据类型一致
# # 有默认参数不输入完整的参数也可以
# data = {
#         "text":"西湖的景色","num_return_sequences":5,
#         "max_length":128,"top_p":0.6
#         }
# r = requests.get(URL,params=data)
# print(r.text)

# --------------------------------- 交互式APi文档 --------------------------------- #
# 访问 http://xx.xx.xx.xx:6605/docs

# --------------------------------- 可选的APi文档 --------------------------------- #
#  http://127.0.0.1:8000/redoc