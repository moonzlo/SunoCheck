"""Upload release files to VirusTotal, wait until every analysis completes and write the results as JSON.

Usage: python virustotal_scan.py <out.json> <file...>
The API key is read from the VT_API_KEY environment variable.
"""

import hashlib
import json
import os
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypedDict, cast

API: str = "https://www.virustotal.com/api/v3"
POLL_SECONDS: int = 30
# Large executables sit in the public queue for a while; past this the release fails instead of shipping without a report.
MAX_WAIT_SECONDS: int = 45 * 60
UPLOAD_TIMEOUT_SECONDS: int = 600
REQUEST_TIMEOUT_SECONDS: int = 60


class AnalysisStats(TypedDict):
    malicious: int
    suspicious: int
    undetected: int
    harmless: int


class AnalysisAttributes(TypedDict):
    status: str
    stats: AnalysisStats


class FileInfo(TypedDict):
    sha256: str


class AnalysisMeta(TypedDict):
    file_info: FileInfo


class AnalysisData(TypedDict):
    attributes: AnalysisAttributes


class AnalysisResponse(TypedDict):
    data: AnalysisData
    meta: AnalysisMeta


@dataclass(frozen=True)
class ScanResult:
    file: str
    sha256: str
    malicious: int
    suspicious: int
    engines: int
    url: str


class MissingApiKeyError(RuntimeError):
    pass


class AnalysisTimeoutError(TimeoutError):
    pass


class HashMismatchError(ValueError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def multipart_body(path: Path, boundary: str) -> bytes:
    head: str = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    )
    return head.encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()


def request_json(request: urllib.request.Request, timeout: int) -> dict[str, object]:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return cast(dict[str, object], json.load(response))


def upload(path: Path, api_key: str) -> str:
    """Returns the analysis id; the upload_url endpoint is used because release executables exceed the 32 MB direct limit."""
    headers: dict[str, str] = {"x-apikey": api_key, "accept": "application/json"}
    url_request = urllib.request.Request(f"{API}/files/upload_url", headers=headers)
    upload_url: str = cast(str, request_json(url_request, REQUEST_TIMEOUT_SECONDS)["data"])
    boundary: str = uuid.uuid4().hex
    request = urllib.request.Request(
        upload_url,
        data=multipart_body(path, boundary),
        headers={**headers, "content-type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return cast(dict[str, str], request_json(request, UPLOAD_TIMEOUT_SECONDS)["data"])["id"]


def wait_for_analysis(analysis_id: str, api_key: str) -> AnalysisResponse:
    request = urllib.request.Request(f"{API}/analyses/{analysis_id}", headers={"x-apikey": api_key, "accept": "application/json"})
    deadline: float = time.monotonic() + MAX_WAIT_SECONDS
    while time.monotonic() < deadline:
        response: AnalysisResponse = cast(AnalysisResponse, request_json(request, REQUEST_TIMEOUT_SECONDS))
        status: str = response["data"]["attributes"]["status"]
        print(f"  analysis {analysis_id}: {status}", flush=True)
        if status == "completed":
            return response
        time.sleep(POLL_SECONDS)
    raise AnalysisTimeoutError(f"VirusTotal analysis {analysis_id} did not complete within {MAX_WAIT_SECONDS} s; re-run the job")


def scan_result(path: Path, local_sha256: str, response: AnalysisResponse) -> ScanResult:
    remote_sha256: str = response["meta"]["file_info"]["sha256"]
    if remote_sha256 != local_sha256:
        raise HashMismatchError(f"VirusTotal analysed {remote_sha256}, but {path.name} is {local_sha256}")
    stats: AnalysisStats = response["data"]["attributes"]["stats"]
    return ScanResult(
        file=path.name,
        sha256=local_sha256,
        malicious=stats["malicious"],
        suspicious=stats["suspicious"],
        engines=stats["malicious"] + stats["suspicious"] + stats["undetected"] + stats["harmless"],
        url=f"https://www.virustotal.com/gui/file/{local_sha256}",
    )


def main(out: Path, files: list[Path]) -> None:
    api_key: str = os.environ.get("VT_API_KEY", "")
    if not api_key:
        raise MissingApiKeyError("VT_API_KEY is empty: add the VIRUSTOTAL secret to the 'deploy' environment")
    results: list[ScanResult] = []
    for path in files:
        local_sha256: str = sha256_of(path)
        print(f"{path.name} {local_sha256}: uploading", flush=True)
        response: AnalysisResponse = wait_for_analysis(upload(path, api_key), api_key)
        result: ScanResult = scan_result(path, local_sha256, response)
        print(f"  {result.malicious} malicious, {result.suspicious} suspicious of {result.engines} engines: {result.url}", flush=True)
        results.append(result)
    out.write_text(json.dumps([asdict(result) for result in results], indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: virustotal_scan.py <out.json> <file...>")
    main(Path(sys.argv[1]), [Path(arg) for arg in sys.argv[2:]])
