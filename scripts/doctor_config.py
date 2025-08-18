import config, json
print("CONFIG PATH:", getattr(config, "__file__", "?"))
keys = ["TTS_MODE","TTS_VOICE","TTS_FORMAT","TTS_RATE","TTS_VOLUME","CHAT_MODEL"]
print(json.dumps({k:getattr(config,k,"<MISSING>") for k in keys}, ensure_ascii=False))
