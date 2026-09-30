"""HTTP-only helpers for the independent service acceptance suite."""

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier


class ServiceError(AssertionError):
    pass


def request(base_url, method, path, body=None, timeout=180):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        status = response.status
        raw = response.read().decode("utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        raise ServiceError(f"{method} {path} returned non-JSON HTTP {status}: {raw[:300]}")
    return {"status": status, "body": payload, "seconds": time.monotonic() - started}


def stream(base_url, path, body, timeout=180, cancel_after_chunks=None, on_chunk=None):
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps({**body, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    chunks = []
    started = time.monotonic()
    done = False
    with urllib.request.urlopen(req, timeout=timeout) as response:
        if response.status != 200:
            raise ServiceError(f"Streaming request returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data))
            if on_chunk is not None:
                on_chunk(chunks[-1])
            if cancel_after_chunks is not None and len(chunks) >= cancel_after_chunks:
                break
    if cancel_after_chunks is None and not done:
        raise ServiceError("Streaming response ended without [DONE]")
    if cancel_after_chunks is not None and not chunks:
        raise ServiceError("Cancellation workload received no stream chunks")
    return {"chunks": chunks, "done": done, "seconds": time.monotonic() - started}


def concurrent_requests(base_url, path, bodies, timeout=180):
    barrier = Barrier(len(bodies))

    def send(body):
        barrier.wait(timeout=timeout)
        return request(base_url, "POST", path, body, timeout)

    with ThreadPoolExecutor(max_workers=len(bodies)) as executor:
        return list(executor.map(send, bodies))
