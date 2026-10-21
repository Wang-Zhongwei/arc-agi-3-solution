"""Download one kernel version's output files (the kaggle CLI ignores /N)."""
import os, re, sys, requests
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

owner, slug, version, out, pattern = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
api = KaggleApi(); api.authenticate()
os.makedirs(out, exist_ok=True)
rx = re.compile(pattern); token = None; n = 0
with api.build_kaggle_client() as kaggle:
    while True:
        req = ApiListKernelSessionOutputRequest()
        req.user_name, req.kernel_slug, req.version_label = owner, slug, version
        if token: req.page_token = token
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
        for item in resp.files or []:
            if not rx.search(item.file_name): continue
            path = os.path.join(out, item.file_name); os.makedirs(os.path.dirname(path), exist_ok=True)
            open(path, "wb").write(requests.get(item.url).content); n += 1
        if resp.log and not os.path.exists(os.path.join(out, f"{slug}.log")):
            open(os.path.join(out, f"{slug}.log"), "w").write(resp.log)
        token = resp.next_page_token
        if not token: break
print(f"v{version}: {n} files -> {out}")
