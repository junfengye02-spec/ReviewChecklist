from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import requests


class PlatformError(RuntimeError):
    pass


class PlatformClient:
    def __init__(self, base_url: str, username: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.session = requests.Session()
        self.token = ""

    def refresh_token(self) -> None:
        url = f"{self.base_url}/public/token.do"
        response = self.session.get(
            url,
            params={"action": "generate", "username": self.username, "userId": "1"},
            timeout=30,
        )
        payload = self._json(response, "generate token")
        if str(payload.get("code")) != "0" or not payload.get("data"):
            raise PlatformError(f"Token generation failed: {payload.get('message') or payload}")
        self.token = str(payload["data"])

    @property
    def auth_headers(self) -> dict[str, str]:
        if not self.token:
            raise PlatformError("Token has not been generated")
        return {"Authorization": f"Bearer {self.token}"}

    def create_product(self, product_name: str, product_property: str) -> dict[str, Any]:
        payload = {
            "productCode": product_name,
            "productName": product_name,
            "productProperty": product_property,
        }
        response = self.session.post(
            f"{self.base_url}/api/basedata/product/create",
            headers={**self.auth_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        data = self._json(response, "create product")
        if data.get("success") is not True or not (data.get("data") or {}).get("id"):
            raise PlatformError(f"Product creation failed: {data.get('message') or data}")
        return {"request": payload, "response": data}

    def import_review_points(self, product_id: str, excel_path: Path) -> dict[str, Any]:
        with excel_path.open("rb") as stream:
            response = self.session.post(
                f"{self.base_url}/api/basedata/reviewpoints/importExcel",
                headers=self.auth_headers,
                data={"productId": product_id},
                files={
                    "file": (
                        excel_path.name,
                        stream,
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                },
                timeout=120,
            )
        data = self._json(response, "import review points")
        if data.get("success") is not True:
            raise PlatformError(f"Review-point import failed: {data.get('message') or data}")
        return data

    def upload_zip(self, product_id: str, zip_path: Path) -> dict[str, Any]:
        with zip_path.open("rb") as stream:
            response = self.session.post(
                f"{self.base_url}/api/ai-review/task/upload",
                headers=self.auth_headers,
                data={"productId": product_id},
                files={"files": (zip_path.name, stream, "application/zip")},
                timeout=600,
            )
        data = self._json(response, "upload ZIP")
        if data.get("success") is not True:
            raise PlatformError(f"ZIP upload failed: {data.get('message') or data}")
        matches = self._matches(data)
        if not data.get("extractResults") or not matches:
            raise PlatformError("ZIP uploaded but no extracted files or review-point matches were returned")
        return data

    def create_task(
        self,
        product_id: str,
        product_name: str,
        applicant_name: str,
        product_property: str,
        clinical_evaluation: str,
        upload: dict[str, Any],
    ) -> dict[str, Any]:
        matches = self._matches(upload)
        payload = {
            "productId": int(product_id),
            "productName": product_name,
            "applicantName": applicant_name,
            "productModel": time.strftime("%Y%m%d%H%M%S"),
            "productProperty": product_property,
            "clinicalEvaluation": clinical_evaluation,
            "extractResults": upload.get("extractResults") or [],
            "matchResult": {"matchedFiles": matches, "matchResults": matches},
        }
        response = self.session.post(
            f"{self.base_url}/api/ai-review/task/create",
            headers={**self.auth_headers, "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        data = self._json(response, "create task")
        task_id = (data.get("data") or {}).get("id")
        if data.get("success") is not True or not task_id:
            raise PlatformError(f"Task creation failed: {data.get('message') or data}")
        return {"request": payload, "response": data, "task_id": str(task_id)}

    def start_task(self, task_id: str) -> dict[str, Any]:
        response = self.session.post(
            f"{self.base_url}/api/ai-review/task/startAiReview/{task_id}",
            headers=self.auth_headers,
            timeout=60,
        )
        data = self._json(response, "start task")
        if data.get("success") is not True:
            raise PlatformError(f"Task start failed: {data.get('message') or data}")
        return data

    def delete_task(self, task_id: str) -> dict[str, Any]:
        response = self.session.delete(
            f"{self.base_url}/api/ai-review/task/delete/{task_id}",
            headers=self.auth_headers,
            timeout=60,
        )
        data = self._json(response, "delete task")
        if data.get("success") is not True:
            raise PlatformError(f"Task deletion failed: {data.get('message') or data}")
        return data

    def wait_for_task(
        self,
        task_id: str,
        interval_seconds: float,
        timeout_seconds: float,
        on_progress=None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_seconds
        last_snapshot: dict[str, Any] = {}
        while time.monotonic() < deadline:
            try:
                response = self.session.post(
                    f"{self.base_url}/api/ai-review/task/progress/batch",
                    headers={**self.auth_headers, "Content-Type": "application/json"},
                    json={"taskIds": [task_id]},
                    timeout=45,
                )
                data = self._json(response, "poll task")
            except requests.RequestException:
                time.sleep(interval_seconds)
                continue
            rows = data.get("data") or []
            if rows:
                last_snapshot = rows[0]
                if on_progress:
                    on_progress(last_snapshot)
                status = str(last_snapshot.get("status") or "").upper()
                percentage = float(last_snapshot.get("percentage") or 0)
                if status in {
                    "FAILED",
                    "ERROR",
                    "CANCELLED",
                    "ABORTED",
                    "AI_REVIEW_FAILED",
                }:
                    raise PlatformError(
                        f"Task ended with status {status}: {last_snapshot}"
                    )
                if status == "COMPLETED" or percentage >= 100:
                    return last_snapshot
            time.sleep(interval_seconds)
        raise TimeoutError(f"Task {task_id} did not complete; last progress={last_snapshot}")

    def get_task_progress(self, task_ids: list[str]) -> list[dict[str, Any]]:
        last_error = ""
        for attempt in range(1, 6):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/ai-review/task/progress/batch",
                    headers={**self.auth_headers, "Content-Type": "application/json"},
                    json={"taskIds": task_ids},
                    timeout=45,
                )
                data = self._json(response, "poll tasks")
                if data.get("success") is False:
                    raise PlatformError(
                        f"Task polling failed: {data.get('message') or data}"
                    )
                rows = data.get("data") or []
                if not isinstance(rows, list):
                    raise PlatformError(
                        "Task polling returned an unexpected data payload"
                    )
                return [row for row in rows if isinstance(row, dict)]
            except (requests.RequestException, PlatformError) as exc:
                last_error = str(exc)
            if attempt < 5:
                time.sleep(2)
        raise PlatformError(f"Task polling failed after retries: {last_error}")

    def get_results(self, task_id: str) -> dict[str, Any]:
        last_error = ""
        for attempt in range(1, 11):
            try:
                response = self.session.get(
                    f"{self.base_url}/api/ai-review/task/aiReviewResults/{task_id}",
                    headers=self.auth_headers,
                    timeout=120,
                )
                data = self._json(response, "get results")
                if data.get("success") is True and data.get("data"):
                    return data
                last_error = str(data.get("message") or data)
            except (requests.RequestException, PlatformError) as exc:
                last_error = str(exc)
            if attempt < 10:
                time.sleep(2)
        raise PlatformError(f"Review report is empty after retries: {last_error}")

    def workflow_test(self, payload: dict[str, Any]) -> Any:
        last_error = ""
        max_attempts = 15
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/workflow/test",
                    headers={**self.auth_headers, "Content-Type": "application/json"},
                    json=payload,
                    timeout=180,
                )
                data = self._json(response, "workflow test")
                output = ((data.get("data") or {}).get("data") or {}).get("output")
                if output not in (None, ""):
                    return output
                last_error = str(data.get("message") or data)
            except (requests.RequestException, PlatformError) as exc:
                last_error = str(exc)
            if attempt < max_attempts:
                time.sleep(2)
        raise PlatformError(f"Workflow test failed after retries: {last_error}")

    @staticmethod
    def _matches(upload: dict[str, Any]) -> list[dict[str, Any]]:
        match_result = upload.get("matchResult") or {}
        return (
            match_result.get("matchResults")
            or match_result.get("matchedFiles")
            or upload.get("matchResults")
            or []
        )

    @staticmethod
    def _json(response: requests.Response, action: str) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as exc:
            preview = (response.text or "")[:500]
            raise PlatformError(
                f"{action} returned non-JSON HTTP {response.status_code}: {preview}"
            ) from exc
        if response.status_code != 200:
            raise PlatformError(
                f"{action} returned HTTP {response.status_code}: {data.get('message') or data}"
            )
        if not isinstance(data, dict):
            raise PlatformError(f"{action} returned an unexpected payload: {type(data).__name__}")
        return data
