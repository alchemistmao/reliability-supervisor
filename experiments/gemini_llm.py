"""Minimal Gemini client (free tier, REST, no extra packages) with the same interface as run_llm_experiments.LLM."""
import getpass, json, os, time, urllib.request, urllib.error
import run_llm_experiments as R

API = "https://generativelanguage.googleapis.com/v1beta"


class GeminiLLM:
    def __init__(self, model=None, min_interval=6.5):
        self.key = os.environ.get("GEMINI_API_KEY") or getpass.getpass("Cole a chave do Gemini: ").strip()
        self.model = model or self.pick_model()
        self.min_interval, self.last = min_interval, 0.0
        self.dry = False

    def _get(self, url):
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.loads(r.read())

    def pick_model(self):
        ms = self._get(f"{API}/models?key={self.key}&pageSize=200").get("models", [])
        names = [m["name"].split("/")[-1] for m in ms if "generateContent" in m.get("supportedGenerationMethods", [])]
        lite = sorted([n for n in names if "flash-lite" in n and "preview" not in n and "exp" not in n], reverse=True)
        if not lite:
            raise RuntimeError(f"no flash-lite model available; models: {names[:20]}")
        return lite[0]

    def cost(self):
        return 0.0

    def call(self, system, user, temperature):
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"maxOutputTokens": 2000}}
        for attempt in range(8):
            wait = self.min_interval - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            req = urllib.request.Request(f"{API}/models/{self.model}:generateContent?key={self.key}",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    j = json.loads(r.read())
                u = j.get("usageMetadata", {})
                with R.LOCK:
                    R.USAGE["in"] += u.get("promptTokenCount", 0); R.USAGE["out"] += u.get("candidatesTokenCount", 0)
                    R.USAGE["calls"] += 1
                parts = j["candidates"][0]["content"].get("parts", [])
                return "".join(p.get("text", "") for p in parts if not p.get("thought"))
            except urllib.error.HTTPError as e:
                msg = e.read().decode()[:300]
                if e.code == 429 and "PerDay" in msg:
                    raise RuntimeError("daily free quota exhausted - run again tomorrow (results are cached)")
                if e.code in (429, 500, 503):
                    time.sleep(30 * (attempt + 1)); continue
                raise RuntimeError(f"Gemini HTTP {e.code}: {msg}")
        raise RuntimeError("Gemini: too many retries")
